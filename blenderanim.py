import math
import bpy, csv, os, random

# === SETTINGS ===
CSV_PATH = r"experiment_data/VelocityVerletSolver(0.001082)_positions.csv"
META_PATH = r"experiment_data/VelocityVerletSolver(0.001082)_metadata.csv"
BASE_NAME = "mass_id"
START_FRAME = None  # If None, starts at frame 1
SCALE_FACTOR = 1  # Visual scale of spheres
CAMERA_HEIGHT_FACTOR = 0.4  # Vertical offset as fraction of diag
CAMERA_RADIUS_FACTOR = 1.2  # Orbit radius as fraction of diag

# === READ METADATA ===
masses = {}
try:
    with open(META_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                masses[row[0]] = float(row[1])
except FileNotFoundError:
    print(f"Warning: Metadata file not found at: {META_PATH}")
    pass

# === READ POSITION DATA ===
data = {}
try:
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for mid, *coords in reader:
            if len(coords) < 3:
                continue
            try:
                pos = tuple(map(float, coords[:3]))
            except (ValueError, IndexError):
                continue
            data.setdefault(mid, []).append(pos)
except FileNotFoundError:
    raise RuntimeError(f"FATAL: Position data file not found at: {CSV_PATH}")

if not data:
    raise RuntimeError("No valid position data loaded from CSV.")

# === VALIDATE FRAMES ===
lengths = {len(v) for v in data.values()}
if len(lengths) > 1:
    print("Warning: Unequal time-series lengths per mass_id. Using shortest length.")
    num_frames = min(lengths)
elif lengths:
    num_frames = lengths.pop()
else:
    raise RuntimeError("No frames to process.")

start_frame = START_FRAME if START_FRAME is not None else 1

# === COMPUTE EFFECTIVE BOUNDING BOX CONSIDERING SPHERE SIZES ===
effective_min = [float("inf")] * 3
effective_max = [float("-inf")] * 3
for mid, positions in data.items():
    if not positions:
        continue
    mass = masses.get(str(mid), 1.0)
    scale = SCALE_FACTOR * (mass ** (1.0 / 3.0))
    min_pos = [min(p[i] for p in positions) for i in range(3)]
    max_pos = [max(p[i] for p in positions) for i in range(3)]
    for i in range(3):
        effective_min[i] = min(effective_min[i], min_pos[i] - scale)
        effective_max[i] = max(effective_max[i], max_pos[i] + scale)

center = [(effective_min[i] + effective_max[i]) / 2 for i in range(3)]
diag = (
    math.sqrt(sum((effective_max[i] - effective_min[i]) ** 2 for i in range(3)))
    if effective_min != effective_max
    else 1.0
)

# === SCENE SETUP ===
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.frame_start = start_frame
scene.frame_end = start_frame + num_frames - 1

# *** Enable Compositor and Set Up Bloom Effect ***
scene.use_nodes = True
node_tree = scene.node_tree
node_tree.nodes.clear()
render_layers = node_tree.nodes.new("CompositorNodeRLayers")
composite = node_tree.nodes.new("CompositorNodeComposite")
glare = node_tree.nodes.new("CompositorNodeGlare")
glare.glare_type = "FOG_GLOW"
glare.threshold = 1.0
glare.size = 2
node_tree.links.new(render_layers.outputs["Image"], glare.inputs["Image"])
node_tree.links.new(glare.outputs["Image"], composite.inputs["Image"])

# Set background to black
world = bpy.data.worlds.get("World")
if world is None:
    world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg_nodes = world.node_tree.nodes
bg_nodes.clear()
bg_output = bg_nodes.new(type="ShaderNodeOutputWorld")
bg_shader = bg_nodes.new(type="ShaderNodeBackground")
bg_shader.inputs["Color"].default_value = (0, 0, 0, 1)
world.node_tree.links.new(bg_shader.outputs["Background"], bg_output.inputs["Surface"])

# Remove existing mesh objects
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="MESH")
bpy.ops.object.delete()

# === CREATE GLOWING SPHERES ===
for mid, positions in data.items():
    positions = positions[:num_frames]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1)
    obj = bpy.context.object
    obj.name = f"{BASE_NAME}_{mid}"

    # Create glowing emission material
    mat = bpy.data.materials.new(name=f"Mat_{mid}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    rng = random.Random(hash(mid))
    color = (rng.random(), rng.random(), rng.random(), 1)
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 5.0
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()

    # Scale by mass
    mass = masses.get(str(mid), 1.0)
    scale = SCALE_FACTOR * (mass ** (1.0 / 3.0))
    obj.scale = (scale, scale, scale)

    # Animate positions
    obj.animation_data_clear()
    action = bpy.data.actions.new(f"{obj.name}_Action")
    obj.animation_data_create()
    obj.animation_data.action = action
    fcurves = [action.fcurves.new(data_path="location", index=i) for i in range(3)]
    for t, loc in enumerate(positions):
        frame = start_frame + t
        for i in range(3):
            fcurves[i].keyframe_points.insert(frame, loc[i], options={"FAST"})
    for fc in fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

# === CAMERA SETUP ===
bpy.ops.object.camera_add(location=(0, 0, 0))
cam = bpy.context.object
cam.name = "Camera"
scene.camera = cam
cam.location = (
    center[0],
    center[1] - diag * CAMERA_RADIUS_FACTOR,
    center[2] + diag * CAMERA_HEIGHT_FACTOR,
)
cam.data.lens = 35
cam.data.clip_start = 0.1
cam.data.clip_end = diag * 10

# Create an empty at the center for the camera to track
bpy.ops.object.empty_add(type="PLAIN_AXES", location=center)
empty = bpy.context.object
empty.name = "Orbit_Center"

track = cam.constraints.new(type="TRACK_TO")
track.target = empty
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"

# Animate camera orbiting the empty
empty.animation_data_clear()
action = bpy.data.actions.new("Orbit_Action")
empty.animation_data_create()
empty.animation_data.action = action
rot_fc = action.fcurves.new(data_path="rotation_euler", index=2)
rot_fc.keyframe_points.insert(start_frame, 0)
rot_fc.keyframe_points.insert(scene.frame_end, -2 * math.pi)
for kp in rot_fc.keyframe_points:
    kp.interpolation = "LINEAR"

# === COLOR MANAGEMENT & RENDER SETTINGS ===
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = -4.0

output_dir = os.path.dirname(CSV_PATH)
scene.render.filepath = os.path.join(output_dir, "rendered_video.mp4")
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS"
scene.render.ffmpeg.ffmpeg_preset = "GOOD"
scene.render.ffmpeg.audio_codec = "NONE"
scene.render.use_file_extension = True
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

print("\n" + "=" * 40)
print(f"Setup complete. Rendering {num_frames} frames.")
print(f"Blender version specific fixes applied.")
print(f"Output will be saved to: {scene.render.filepath}")
print("=" * 40 + "\n")

bpy.ops.render.render(animation=True)
