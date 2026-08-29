#!/usr/bin/env python3
"""
Node where the ollama language model lives. It receives plain English instructions and returns a JSON array of tasks. 
The executor then executes those tasks.
planner_node -- this is where the language model lives.

An ordinary ROS 2 node. The only unusual thing is that instead of reading a
sensor, it makes an HTTP POST to a server on localhost.

Subscribes : /instruction  (std_msgs/String)  plain English
Publishes  : /task_plan    (std_msgs/String)  JSON array of tasks

Uses urllib from the standard library rather than the ollama package. The
request below is the same HTTP call you already made with curl, so there is
no library hiding the mechanism.
"""

import json
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

# --------------------------------------------------------------------------
# The schema. Identical to the one used in the curl step. Ollama compiles it
# into a grammar and masks any token that would violate it during decoding.
# --------------------------------------------------------------------------

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "pick", "place", "inspect", "home"],
                    },
                    "target": {"type": "string"},
                    "destination": {"type": "string"},
                    # The operator's own words for the target, copied
                    # verbatim. The robot grounds THIS, not the translation.
                    #
                    # Without it the planner resolves ambiguity before the
                    # grounder can object: "the greenish shape" became
                    # "green cylinder", an exact scene name, so the executor
                    # matched it perfectly and never noticed that five green
                    # objects existed. Carrying the original words forward
                    # lets the existing ambiguity check do its job.
                    "said": {"type": "string"},
                },
                # destination is required for EVERY task, not
                # just place. The schema cannot express "only
                # when action is place", and asking in the
                # prompt did not hold: once "said" was added
                # the model started dropping destination and
                # every place became invalid. Requiring the key
                # makes the grammar emit it. Non place tasks
                # just leave it empty and it is ignored.
                "required": ["action", "target", "said",
                             "destination"],
            },
        }
    },
    "required": ["tasks"],
}

# The grammar fixes the shape. This fixes the meaning. Both are needed.
# The grammar fixes the shape of the output. The prompt fixes its meaning.
# An example does far more work here than a written rule: measured on the same
# machine, three rules as prose produced 92 tokens with the target and
# destination fields swapped, while one worked example produced 46 tokens and
# was correct.
BASE_PROMPT = """You are a task planner for a robot arm on a table.
Every pick must be preceded by a navigate to the same target.
In a place task, 'target' is the object being carried and 'destination' is the
surface it goes onto.

Emit tasks only, no commentary."""

VALID_ACTIONS = {"navigate", "pick", "place", "inspect", "home"}

# Words that mean the operator actually asked to return to the rest pose.
HOME_INTENT = {"home", "rest", "back", "return", "returning", "retract",
               "stow", "park", "reset", "neutral", "origin", "start"}


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("planner_node")

        self.declare_parameter("model", "qwen3.5:4b")
        self.declare_parameter("host", "http://localhost:11434")
        self.declare_parameter("timeout_s", 120.0)

        self.model = self.get_parameter("model").value
        self.host = self.get_parameter("host").value.rstrip("/")
        self.timeout = float(self.get_parameter("timeout_s").value)

        # Fail at startup rather than on the first instruction. A node that
        # starts fine and dies mid demo is much worse than one that refuses
        # to start.
        self._preflight()

        self.objects = []
        self.destinations = []
        self.colors = []
        self.shapes = []
        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(
            String, "scene_objects", self.on_objects, latched)

        self.pub = self.create_publisher(String, "task_plan", 10)
        self.create_subscription(String, "instruction", self.on_instruction, 10)

        self.get_logger().info(f"planner ready, model={self.model}")
        self.get_logger().info(
            "try: ros2 topic pub --once /instruction std_msgs/String "
            "\"{data: 'pick up the red item and put it on the tray'}\""
        )

    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _preflight(self) -> None:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as r:
                tags = json.loads(r.read().decode())
        except Exception as exc:
            self.get_logger().fatal(
                f"Cannot reach Ollama at {self.host}. "
                f"Is the service running? ({exc})"
            )
            raise SystemExit(1)

        installed = [m["name"] for m in tags.get("models", [])]
        if self.model not in installed:
            self.get_logger().fatal(
                f"Model '{self.model}' is not pulled. "
                f"Run: ollama pull {self.model}\nInstalled: {installed}"
            )
            raise SystemExit(1)

    # ------------------------------------------------------------------

    def on_objects(self, msg: String) -> None:
        """The executor tells us what is actually in the scene. Injecting the
        list means the model uses real names instead of inventing plausible
        ones, so a mismatch is refused here rather than three nodes later."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        new = data.get("objects", [])
        if new and new != self.objects:
            self.objects = new
            self.destinations = data.get("destinations", [])
            self.colors = data.get("colors", [])
            self.shapes = data.get("shapes", [])
            self.get_logger().info(
                f"scene: {len(new)} objects, "
                f"colours [{', '.join(self.colors)}], "
                f"shapes [{', '.join(self.shapes)}]")

    def system_prompt(self) -> str:
        """Describe the world, then ask the model to TRANSLATE into it.

        The earlier version of this prompt said "copy the words the operator
        used". That was backwards. Measured behaviour: with a short sentence
        the model translated "blue coke can" to "blue cylinder" correctly,
        but with a longer sentence the copy instruction won and it passed the
        phrase through verbatim, which then failed grounding.

        Translation is what a language model is good at. Doing it here means
        the executor does not need a synonym table for every word a person
        might invent.
        """
        if not self.objects:
            return BASE_PROMPT

        parts = [BASE_PROMPT, ""]
        if self.colors and self.shapes:
            parts.append(
                f"Every object in the scene is named "
                f"'<colour> <shape>'.\n"
                f"Colours: {', '.join(self.colors)}.\n"
                f"Shapes: {', '.join(self.shapes)}.\n"
                f"Cubes also take a size, small or large, so a cube is named "
                f"'<size> <colour> cube'."
            )
        parts.append(
            "TRANSLATE whatever the operator says into one of these names. "
            "Never pass their wording through unchanged.\n"
            "A can, tin, bottle, cup or container is a cylinder.\n"
            "A ball, marble or orb is a sphere.\n"
            "A brick, bar, slab or plank is a cuboid.\n"
            "A block, box or square is a cube.\n"
            "Brand names and materials are irrelevant. A coke can is a "
            "cylinder. A lego block is a cube."
        )
        if self.destinations:
            dest_line = (f"Places to put things: "
                         f"{', '.join(self.destinations)}. "
                         f"These are never picked up.")
            if "table" in self.destinations:
                dest_line += (
                    " The table is the surface everything stands on: a "
                    "desk, workbench, counter, worktop or surface means "
                    "the table. Putting an object back, or back where it "
                    "was, means placing it on the table. That is a place "
                    "task, not a home task.")
            parts.append(dest_line)

        parts.append(
            "MOVING AN OBJECT ALWAYS TAKES FOUR TASKS.\n"
            "  navigate to the object\n"
            "  pick it up\n"
            "  navigate to where it is going\n"
            "  place it there\n"
            "\n"
            "This holds every single time an object moves. If the operator "
            "moves the same object twice, that is eight tasks, not five. The "
            "arm has one gripper: it cannot pick up a second thing while "
            "still holding the first, and it must never finish a plan still "
            "carrying something. A plan that picks up without placing is "
            "rejected.")

        # Two examples on purpose. A single example that ends in a home task
        # teaches the model that every plan ends that way, and it starts
        # appending home unprompted. The pair shows that home appears only
        # when the operator asks for it.
        parts.append(
            'Example 1. The operator did not mention going home, so there is '
            'no home task.\n'
            'Instruction: hey, grab the blue coke can and stick it on the '
            'shelf\n'
            'Output: {"tasks":['
            '{"action":"navigate","target":"blue cylinder",'
            '"said":"blue coke can","destination":""},'
            '{"action":"pick","target":"blue cylinder",'
            '"said":"blue coke can","destination":""},'
            '{"action":"navigate","target":"shelf","said":"shelf",'
            '"destination":""},'
            '{"action":"place","target":"blue cylinder",'
            '"said":"blue coke can","destination":"shelf"}]}'
        )
        parts.append(
            'Example 2. The operator asked to go home, so one home task is '
            'added at the end.\n'
            'Instruction: put the red brick on the tray then return to '
            'your home position\n'
            'Output: {"tasks":['
            '{"action":"navigate","target":"red cuboid","said":"red brick",'
            '"destination":""},'
            '{"action":"pick","target":"red cuboid","said":"red brick",'
            '"destination":""},'
            '{"action":"navigate","target":"tray","said":"tray",'
            '"destination":""},'
            '{"action":"place","target":"red cuboid","said":"red brick",'
            '"destination":"tray"},'
            '{"action":"home","target":"home","said":"home",'
            '"destination":""}]}'
        )
        parts.append(
            'Example 3. The operator named no shape. Note that "said" stays '
            'vague. Do NOT write "green cylinder" there.\n'
            'Instruction: put the greenish thing on the tray\n'
            'Output: {"tasks":['
            '{"action":"navigate","target":"green cylinder",'
            '"said":"greenish thing","destination":""},'
            '{"action":"pick","target":"green cylinder",'
            '"said":"greenish thing","destination":""},'
            '{"action":"navigate","target":"tray","said":"tray",'
            '"destination":""},'
            '{"action":"place","target":"green cylinder",'
            '"said":"greenish thing","destination":"tray"}]}'
        )
        parts.append(
            'Example 4. The same object is moved twice, so there are two '
            'complete pick and place cycles.\n'
            'Instruction: put the red brick on the shelf, then move it to '
            'the tray\n'
            'Output: {"tasks":['
            '{"action":"navigate","target":"red cuboid","said":"red brick",'
            '"destination":""},'
            '{"action":"pick","target":"red cuboid","said":"red brick",'
            '"destination":""},'
            '{"action":"navigate","target":"shelf","said":"shelf",'
            '"destination":""},'
            '{"action":"place","target":"red cuboid","said":"red brick",'
            '"destination":"shelf"},'
            '{"action":"navigate","target":"shelf","said":"shelf",'
            '"destination":""},'
            '{"action":"pick","target":"red cuboid","said":"red brick",'
            '"destination":""},'
            '{"action":"navigate","target":"tray","said":"tray",'
            '"destination":""},'
            '{"action":"place","target":"red cuboid","said":"red brick",'
            '"destination":"tray"}]}'
        )
        # Only taught when the table exists as a place, so the model is
        # never shown an example it cannot ground.
        if "table" in self.destinations:
            parts.append(
                'Example 5. Going somewhere without moving an object is a '
                'single navigate task. Nothing is picked up, so nothing is '
                'placed.\n'
                'Instruction: go to the table\n'
                'Output: {"tasks":['
                '{"action":"navigate","target":"table","said":"table",'
                '"destination":""}]}'
            )
        parts.append(
            "Emit a home task ONLY when the operator explicitly asks to go "
            "home, return, or go back. If they do not say so, the plan must "
            "not contain a home task. Never emit a navigate to home."
        )
        return "\n\n".join(parts)

    def on_instruction(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return

        log = self.get_logger().info
        log("=" * 62)
        log(f'INSTRUCTION  "{text}"')
        log(f"THINKING     {self.model}, {len(self.objects)} objects in scene "
            f"({', '.join(self.colors)} x {', '.join(self.shapes)})")

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,          # Qwen reasons by default. Costs 8x tokens.
            "options": {"temperature": 0, "seed": 42},
            "format": TASK_SCHEMA,   # the grammar constraint
            "messages": [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": text},
            ],
        }

        try:
            resp = self._post("/api/chat", payload, self.timeout)
        except urllib.error.URLError as exc:
            self.get_logger().error(f"inference failed: {exc}")
            return
        except Exception as exc:
            self.get_logger().error(f"unexpected error: {exc}")
            return

        raw = resp.get("message", {}).get("content", "")
        took = resp.get("total_duration", 0) / 1e9
        tokens = resp.get("eval_count", 0)

        tasks = self._validate(raw)
        if tasks is None:
            return

        tasks = self._enforce_home_intent(tasks, text)
        if not tasks:
            self.get_logger().warn("nothing left to do after filtering")
            return

        if not self._check_carry_balance(tasks):
            return

        log = self.get_logger().info
        log(f"PLAN     {len(tasks)} tasks, {tokens} tokens, {took:.1f}s")
        for i, t in enumerate(tasks, 1):
            said = t.get("said", "")
            line = f"  {i}  {t['action']:<9} {t['target']}"
            if (t.get("destination") or "").strip():
                line += f"  ->  {t['destination']}"
            if said and said.lower().strip() != t["target"].lower().strip():
                line += f'      (you said "{said}")'
            log(line)

        out = String()
        out.data = json.dumps({"tasks": tasks})
        self.pub.publish(out)

    def _enforce_home_intent(self, tasks, instruction):
        """Drop home tasks the operator never asked for.

        The prompt asks the model not to add these. It mostly complies, but
        an instruction that must always hold does not belong in a prompt: a
        prompt is a strong suggestion, and this is a rule. Enforcing it here
        makes the behaviour deterministic regardless of model, version or
        phrasing.

        The same argument applies to the grammar. Structure is enforced by
        the decoder because it must never vary. This is the policy equivalent.
        """
        words = set(instruction.lower().replace("?", " ")
                    .replace(",", " ").replace(".", " ").split())
        if words & HOME_INTENT:
            return tasks

        kept = [t for t in tasks if t.get("action") != "home"]
        dropped = len(tasks) - len(kept)
        if dropped:
            self.get_logger().info(
                f"dropped {dropped} unrequested home task"
                f"{'s' if dropped > 1 else ''} "
                f"(operator did not ask to return home)")
        return kept

    def _check_carry_balance(self, tasks):
        """Every pick must have a matching place.

        The arm has one gripper and it either holds something or it does not.
        A plan that picks a second object without releasing the first, or
        that finishes still carrying something, describes a robot that does
        not exist.

        This is checked here rather than asked for in the prompt, for the
        same reason as the other rules that moved into code: a prompt is a
        strong suggestion and this is arithmetic. The failure it catches is
        real. Asked to put an object on the shelf and then move it to the
        tray, the model produced navigate, pick, navigate, place, navigate,
        pick, and then went home still holding it.
        """
        held = None
        for i, t in enumerate(tasks, 1):
            action = t.get("action")
            target = t.get("target", "")

            if action == "pick":
                if held is not None:
                    self.get_logger().error(
                        f'task {i} picks up "{target}" while still holding '
                        f'"{held}". The arm has one gripper.')
                    self.get_logger().error(
                        "  the plan is missing a place. Try saying where "
                        "each object should go.")
                    return False
                held = target

            elif action == "place":
                if held is None:
                    self.get_logger().error(
                        f"task {i} places something, but nothing has been "
                        f"picked up.")
                    return False
                held = None

        if held is not None:
            self.get_logger().error(
                f'this plan picks up "{held}" and never puts it down.')
            self.get_logger().error(
                "  say where it should end up, for example "
                '"...and leave it on the tray".')
            return False

        return True

    def _validate(self, raw: str):
        """Belt and braces. The grammar makes this nearly redundant, which is
        exactly why it stays. If it ever fires, something important changed."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"model returned non-JSON: {exc}")
            self.get_logger().error(f"raw: {raw[:200]}")
            return None

        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            self.get_logger().warn("model returned an empty plan")
            return None

        for i, t in enumerate(tasks, 1):
            if not isinstance(t, dict):
                self.get_logger().error(f"task {i} is not an object")
                return None
            if t.get("action") not in VALID_ACTIONS:
                self.get_logger().error(
                    f"task {i} has invalid action {t.get('action')!r}")
                return None
            if not t.get("target"):
                self.get_logger().error(f"task {i} has no target")
                return None
            # A place with no destination would silently drop the object
            # onto whatever the target resolved to. The schema cannot make
            # this conditional, so it is checked here.
            if (t.get("action") == "place"
                    and not (t.get("destination") or "").strip()):
                self.get_logger().error(
                    f"task {i} is a place with no destination. "
                    f"Nowhere to put it.")
                return None

        return tasks


def main() -> None:
    rclpy.init()
    try:
        node = PlannerNode()
    except SystemExit:
        rclpy.try_shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
