#!/usr/bin/env python3
"""
executor_node -- the bridge between language and motion.

Subscribes : /task_plan      (std_msgs/String)           JSON from the planner
             /world_state    (std_msgs/String, latched)  what is on the table
Publishes  : /arm_goal       (geometry_msgs/PoseStamped) consumed by arm_node
             /arm_home       (std_msgs/Empty)
             /scene_command  (std_msgs/String)           the world changed
             /scene_objects  (std_msgs/String, latched)  names for the planner

This node does three things and knows nothing else.

  1  It receives the world on /world_state. It does not know or care whether
     that world was made up by scene_node or measured by a camera. That
     ignorance is deliberate and is what makes Phase 2 a swap rather than a
     rewrite.

  2  It grounds each task: turns the words the operator used into one object
     in that world, or refuses and says why.

  3  It sends the arm somewhere, and reports back what changed.
"""

import json
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Empty, String

# The world is no longer built here. It arrives on /world_state from
# whichever node is providing it: scene_node in Phase 1, perception_node in
# Phase 2. This file does not know or care which, and that is the point.

APPROACH_HEIGHT = 0.12
PLACE_HEIGHT = 0.08
SETTLE_SECONDS = 2.5
HOME_SECONDS = 3.0

WORLD_FRAME = "world"


# ===========================================================================
# GROUNDING
# The synonym tables that used to live here are gone. Matching a phrase to an
# object is now done by comparing embeddings, in grounding.py. Nobody has to
# anticipate the word "coke".
#
# What remains exact, and only because it must be: size words and rank words.
# See the note at the top of grounding.py.
# ===========================================================================

from agentic_arm.grounding import (  # noqa: E402
    DEFAULT_MARGIN, DEFAULT_MODEL, DEFAULT_THRESHOLD, EmbeddingError,
    Grounder, is_home,
)


# ===========================================================================

class ExecutorNode(Node):
    def __init__(self):
        super().__init__("executor_node")

        # Reshuffle the table without editing code:
        #   ros2 launch agentic_arm full.launch.py scene_seed:=42
        self.declare_parameter("return_home", False)
        self.return_home = bool(self.get_parameter("return_home").value)

        self.declare_parameter("embed_model", DEFAULT_MODEL)
        self.declare_parameter("embed_host", "http://localhost:11434")
        self.declare_parameter("similarity_threshold", DEFAULT_THRESHOLD)
        self.declare_parameter("similarity_margin", DEFAULT_MARGIN)
        self.declare_parameter("augment_descriptions", True)

        # The world arrives on a topic. Until it does, this node has nothing
        # to ground against and refuses to act, which is honest: a robot that
        # cannot see anything should not be executing plans.
        self.world = {}
        self.colors = []
        self.shapes = []
        self.grounder = None

        self.pub_goal = self.create_publisher(PoseStamped, "arm_goal", 10)
        self.pub_home = self.create_publisher(Empty, "arm_home", 10)
        self.pub_command = self.create_publisher(String, "scene_command", 10)
        self.create_subscription(String, "task_plan", self.on_plan, 10)

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.pub_objects = self.create_publisher(
            String, "scene_objects", latched)
        self.create_subscription(
            String, "world_state", self.on_world, latched)

        self.held = None
        self.busy = False

        self.get_logger().info(
            "executor ready, waiting for /world_state")

    # ------------------------------------------------------------------

    def on_plan(self, msg):
        if self.grounder is None:
            self.get_logger().error(
                "no world yet. Nothing is publishing /world_state, so there "
                "is nothing to ground against. Ignoring this plan.")
            return
        if self.busy:
            self.get_logger().warn("still executing, ignoring new plan")
            return
        try:
            tasks = json.loads(msg.data)["tasks"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self.get_logger().error(f"bad plan: {exc}")
            return

        # Run on a worker thread. Blocking inside this callback would stop the
        # node spinning, so no timer ticks and no /tf, and a carried object
        # would freeze on the table instead of following the gripper.
        self.busy = True
        threading.Thread(target=self._run_guarded, args=(tasks,),
                         daemon=True).start()

    def _run_guarded(self, tasks):
        try:
            self.run(tasks)
        finally:
            self.busy = False

    def run(self, tasks):
        total = len(tasks)
        log = self.get_logger().info
        started = time.time()

        log("-" * 62)
        log(f"EXECUTING  {total} task(s)   |   holding: "
            f"{self.held or 'nothing'}")

        for i, task in enumerate(tasks, 1):
            action = task.get("action")
            target = task.get("target", "")
            said = task.get("said", "")
            destination = (task.get("destination") or "").strip()

            header = f"TASK {i}/{total}  {action} {target}"
            if destination:
                header += f" -> {destination}"
            log(header)

            # A home task that names a real object is not a home task.
            # "come back to the tray" contains the word back, which reads as
            # home intent, but the operator named a destination. Going to the
            # rest pose instead would be the opposite of what they asked.
            names_a_place = (
                action == "home"
                and not is_home(target)
                and self.grounder.resolve(target)[0] is not None)
            if names_a_place:
                log(f'  note     "{target}" is a place, not the rest pose. '
                    f"Treating this as a move, not a homing.")
                action = "navigate"

            if action == "home" or is_home(target) or is_home(said):
                log("  goal     rest pose, joint space, no IK needed")
                self.go_home()
                self.sleep(HOME_SECONDS)
                continue

            # A place with no destination is incoherent. Without this the
            # code fell through to grounding the object's own name and
            # "placed" it on itself, leaving it hanging in mid air at its
            # own old position. Fail loudly instead.
            if action == "place" and not destination:
                self.get_logger().error(
                    f'  NO DESTINATION: cannot place "{target}" because the '
                    f'plan does not say where. Aborting.')
                return

            # TWO INDEPENDENT READINGS, and the disagreement is the signal.
            #
            # The planner read the whole sentence and translated it. The
            # grounder reads the operator's own words in isolation. Each
            # fails in a different way:
            #
            #   grounding only the translation
            #       "the greenish thing" silently became green cylinder.
            #       The planner picked one of five and nothing objected.
            #
            #   grounding only the operator's words
            #       "pepsi can that looks like the color of emerald stone"
            #       is ten words of noise around one colour cue. The sentence
            #       embedding muddied it and landed on golden cylinder, while
            #       the planner had already got green cylinder right.
            #
            # So do both. If they agree, act. If the operator's phrase was
            # not decisive and the planner simply chose from the candidates,
            # that is the case to stop and ask about. If they disagree but
            # both are plausible, trust the planner, which saw the full
            # sentence, and say plainly that they differed.
            # For a place, the phrase to resolve is the DESTINATION, and
            # the planner's reading of a destination is the destination
            # itself: there is no separate translation to cross check
            # against. Passing target here compared "tray" against the
            # object being carried, they disagreed, and the object was
            # placed on itself.
            if action == "place":
                phrase = planner_reading = destination
            else:
                phrase = said or target
                planner_reading = target

            key, reason, note = self._ground(phrase, planner_reading, log)
            if key is None:
                return

            # Placing on the TABLE means returning the carried object to
            # where it started, not to the table's centre. The origin comes
            # from the world, so a world without origins (Phase 2: a camera
            # cannot know where things were before it started looking) falls
            # back to the centre and nothing breaks.
            drop_at = None
            if (action == "place"
                    and self.world[key].get("shape") == "table"
                    and self.held is not None):
                origin = (self.world.get(self.held) or {}).get("origin")
                if origin:
                    drop_at = [float(v) for v in origin]

            if drop_at is not None:
                x, y, z = drop_at
                z += PLACE_HEIGHT
                why = "  (back to its starting spot on the table)"
            else:
                x, y, z = self.world[key]["xyz"]
                why = ""
                if action in ("navigate", "inspect"):
                    z += APPROACH_HEIGHT
                    why = f"  (+{int(APPROACH_HEIGHT * 100)}cm approach)"
                elif action == "place":
                    z += PLACE_HEIGHT
                    why = f"  (+{int(PLACE_HEIGHT * 100)}cm above surface)"

            log(f"  goal     x {x:+.3f}  y {y:+.3f}  z {z:+.3f}{why}")

            self.send_goal(x, y, z)
            self.sleep(SETTLE_SECONDS)

            if action == "pick":
                self.do_pick(key)
            elif action == "place":
                self.do_place(key, at=drop_at)

        if self.return_home:
            log("RETURNING HOME")
            self.go_home()
            self.sleep(HOME_SECONDS)

        log(f"DONE     {total} task(s) in {time.time() - started:.1f}s   |   "
            f"holding: {self.held or 'nothing'}")

    # ------------------------------------------------------------------

    def _ground(self, phrase, target, log):
        """Resolve a phrase to one object, cross checked against the planner.

        There are two readings of every target. The planner read the whole
        sentence and translated it into scene vocabulary. The grounder reads
        the operator's own words in isolation. They fail differently:

          the planner    is good at language in context. It correctly turns
                         "the little bottle that looks like the sky" into
                         "blue cylinder".

          the grounder   is good at single phrases and unknown words. It
                         handles "flask" and "maroon" and typos, which the
                         planner may pass through unchanged. But a colour cue
                         buried in eight other words gets diluted, and it
                         lands somewhere wrong with confidence.

        So the rule is: WHEN THE PLANNER NAMES A REAL OBJECT, TRUST IT, and
        say so in the log. It saw more context than the grounder did.

        The grounder earns the decision in exactly two cases. When the
        planner's name is ambiguous, it chooses within what the planner
        narrowed to. When the planner's name matches nothing at all, it is
        the only reading left.

        An earlier version refused whenever the operator's own words were not
        decisive on their own. That rejected "the little bottle that looks
        like the sky", where the planner was plainly right and only the
        embedding was confused. There is no reliable signal separating a
        vague operator from a diluted embedding, so refusing on that basis
        was guesswork dressed up as caution.

        Returns (name, reason, note). name is None if the caller should
        abort; any explanation has already been logged.
        """
        by_planner, _, _ = self.grounder.resolve(target)
        by_words, reason, note = self.grounder.resolve(phrase)

        # 1. The planner named a real object. It read the whole sentence, so
        #    it gets the decision.
        if by_planner is not None:
            if by_words == by_planner:
                log(f'  ground   "{phrase}" -> {by_planner}   '
                    f'({note or "exact name"})')
            elif by_words is not None:
                log(f'  ground   "{phrase}" -> {by_planner}')
                log(f'           your wording alone read as {by_words}; '
                    f"the planner saw the full sentence, using its reading")
            else:
                log(f'  ground   "{phrase}" -> {by_planner}   '
                    f"(the planner's reading of your sentence)")
            return by_planner, None, "planner"

        # 2. The planner's name is ambiguous. Use it to narrow, and let the
        #    operator's wording choose inside that. refines() is exact word
        #    containment, not similarity: "blue cube" gives the two blue
        #    cubes and never a green one.
        shortlist = self.grounder.refines(target)
        if len(shortlist) > 1:
            ranked = self.grounder.rank_within(phrase, shortlist)
            if ranked:
                top_score, top_name = ranked[0]
                clear = (len(ranked) == 1
                         or top_score - ranked[1][0] >= self.grounder.margin)
                if clear:
                    log(f'  ground   "{phrase}" -> {top_name}   '
                        f'(planner narrowed to {len(shortlist)}, '
                        f'your wording chose, {top_score:.2f})')
                    return top_name, None, "narrowed by planner"

            log(f'  ground   the planner read this as "{target}", which '
                f"matches {len(shortlist)} objects")
            log(f"           {', '.join(sorted(shortlist))}")
            self.get_logger().error(
                "  UNCLEAR: nothing you said chooses between them. "
                "Aborting plan.")
            return None, f'"{target}" is ambiguous', None

        # 3. The planner named nothing usable. The operator's words are the
        #    only reading left.
        if by_words is not None:
            log(f'  ground   "{phrase}" -> {by_words}   '
                f'({note or "on your wording"})')
            log(f'           the planner\'s "{target}" matched nothing')
            return by_words, None, note

        # 4. Neither reading found anything.
        detail = reason or (
            f'neither your wording nor the planner\'s "{target}" '
            f"matched anything in the scene")
        self.get_logger().error(f"  GROUND FAILED: {detail}")
        return None, detail, None

    def go_home(self):
        """Ask the arm for its rest pose. No coordinate is sent, because home
        is defined in joint space and needs no inverse kinematics."""
        if self.held is not None:
            self.get_logger().warn(
                f'  homing while carrying "{self.held}", it comes along')
        self.pub_home.publish(Empty())

    def do_pick(self, key):
        if key is None:
            return
        if not self.world[key]["graspable"]:
            self.get_logger().warn(
                f'  "{key}" is furniture, the arm cannot carry it')
            return
        if self.held is not None:
            self.get_logger().warn(
                f'  already carrying "{self.held}", it will be dropped')
        was = self.world[key]["xyz"]
        self.held = key
        self.tell_world(held=key)
        self.get_logger().info(
            f"  HELD     {key}   (was at x {was[0]:+.3f} y {was[1]:+.3f}), "
            f"now follows the gripper")

    def do_place(self, destination_key, at=None):
        if self.held is None:
            self.get_logger().warn("  nothing is being carried, place ignored")
            return
        released = self.held
        self.held = None

        # We do not move the object ourselves any more. We report what
        # happened and whoever owns the world decides what that means. In
        # Phase 2 the camera will simply see it in its new place.
        # "at" is the exact resting spot when we chose one (returning an
        # object to its origin); without it, the world computes the rest
        # position from the destination itself.
        place = {"object": released, "on": destination_key}
        if at is not None:
            place["at"] = at
        self.tell_world(held=None, place=place)
        self.get_logger().info(
            f"  PLACED   {released} on {destination_key}")

    def send_goal(self, x, y, z):
        msg = PoseStamped()
        msg.header.frame_id = WORLD_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)

    def sleep(self, seconds):
        """Plain sleep. Runs on a worker thread, so the main thread keeps
        spinning and the marker timer and /tf listener stay live."""
        time.sleep(seconds)

    def on_world(self, msg):
        """The world arrived, or changed.

        This is the seam. In Phase 1 the sender is scene_node, which makes
        the world up. In Phase 2 it is perception_node, which measures it
        with a camera. Nothing below this line can tell the difference.
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("world_state is not valid JSON")
            return

        objects = data.get("objects") or {}
        if not objects:
            self.get_logger().warn("world_state arrived empty")
            return

        first = self.grounder is None
        self.world = objects
        self.colors = data.get("colors", [])
        self.shapes = data.get("shapes", [])

        if first:
            try:
                self.grounder = Grounder(
                    self.world,
                    host=self.get_parameter("embed_host").value,
                    model=self.get_parameter("embed_model").value,
                    threshold=float(
                        self.get_parameter("similarity_threshold").value),
                    margin=float(
                        self.get_parameter("similarity_margin").value),
                    augment=bool(
                        self.get_parameter("augment_descriptions").value),
                    log=self.get_logger().info)
            except EmbeddingError as exc:
                self.get_logger().fatal(str(exc))
                self.get_logger().fatal(
                    f"Run: ollama pull "
                    f"{self.get_parameter('embed_model').value}")
                raise SystemExit(1)

            graspable = sum(1 for o in self.world.values() if o["graspable"])
            self.get_logger().info(
                f"world received: {len(self.world)} objects, "
                f"{graspable} of them graspable")
            self.publish_object_list()
        else:
            # Positions change often and need no new vectors. Names changing
            # is rare and does.
            if self.grounder.set_scene(self.world):
                self.get_logger().info("object names changed, re-embedded")
                self.publish_object_list()

    def publish_object_list(self):
        """Tell the planner what exists, so it can name things correctly."""
        msg = String()
        msg.data = json.dumps({
            "objects": [n for n, o in self.world.items() if o["graspable"]],
            "destinations": [n for n, o in self.world.items()
                             if not o["graspable"]],
            "colors": self.colors,
            "shapes": self.shapes,
        })
        self.pub_objects.publish(msg)

    def tell_world(self, **changes):
        """Report a change we made, for whoever is providing the world.

        scene_node applies it. A perception node ignores it, because a camera
        does not need telling that an object moved: it sees it.
        """
        msg = String()
        msg.data = json.dumps(changes)
        self.pub_command.publish(msg)


def main():
    rclpy.init()
    try:
        node = ExecutorNode()
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
