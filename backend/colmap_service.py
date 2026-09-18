import os
import subprocess


# ============================================================
# COLMAP PATH
# ============================================================

COLMAP_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "tools",
        "01 COLMAP",
        "colmap-x64-windows-nocuda"
    )
)

COLMAP_EXE = os.path.join(
    COLMAP_ROOT,
    "bin",
    "colmap.exe"
)

QT_PLUGIN_PATH = os.path.join(
    COLMAP_ROOT,
    "plugins"
)


# ============================================================
# CHECK COLMAP
# ============================================================

def check_colmap():

    if not os.path.exists(COLMAP_EXE):
        raise FileNotFoundError(
            f"COLMAP executable not found:\n{COLMAP_EXE}"
        )

    if not os.path.exists(QT_PLUGIN_PATH):
        raise FileNotFoundError(
            f"COLMAP Qt plugins not found:\n{QT_PLUGIN_PATH}"
        )

    return True


# ============================================================
# RUN COLMAP COMMAND
# ============================================================

def run_command(command):

    print("\n========================================")
    print("Running COLMAP command:")
    print(" ".join(command))
    print("========================================")

    env = os.environ.copy()

    # Qt plugin paths
    env["QT_PLUGIN_PATH"] = QT_PLUGIN_PATH

    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        QT_PLUGIN_PATH,
        "platforms"
    )

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return result


# ============================================================
# SPARSE RECONSTRUCTION
# ============================================================

def run_colmap_reconstruction(
    image_folder,
    workspace_folder
):

    check_colmap()

    if not os.path.exists(image_folder):
        raise FileNotFoundError(
            f"Image folder not found:\n{image_folder}"
        )

    os.makedirs(
        workspace_folder,
        exist_ok=True
    )

    # Database
    database_path = os.path.join(
        workspace_folder,
        "database.db"
    )

    # Sparse folder
    sparse_folder = os.path.join(
        workspace_folder,
        "sparse"
    )

    os.makedirs(
        sparse_folder,
        exist_ok=True
    )

    # ========================================================
    # STEP 1 - FEATURE EXTRACTION
    # ========================================================

    feature_command = [
        COLMAP_EXE,
        "feature_extractor",
        "--database_path",
        database_path,
        "--image_path",
        image_folder
    ]

    run_command(feature_command)

    # ========================================================
    # STEP 2 - FEATURE MATCHING
    # ========================================================

    match_command = [
        COLMAP_EXE,
        "exhaustive_matcher",
        "--database_path",
        database_path
    ]

    run_command(match_command)

    # ========================================================
    # STEP 3 - SPARSE RECONSTRUCTION
    # ========================================================

    mapper_command = [
        COLMAP_EXE,
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        image_folder,
        "--output_path",
        sparse_folder
    ]

    run_command(mapper_command)

    return {
        "database_path": database_path,
        "sparse_path": sparse_folder
    }


# ============================================================
# DENSE RECONSTRUCTION
# ============================================================

def run_dense_reconstruction(
    image_folder,
    workspace_folder
):

    check_colmap()

    # --------------------------------------------------------
    # Sparse model
    # --------------------------------------------------------

    sparse_folder = os.path.join(
        workspace_folder,
        "sparse",
        "0"
    )

    if not os.path.exists(sparse_folder):
        raise FileNotFoundError(
            f"Sparse reconstruction not found:\n{sparse_folder}"
        )

    # --------------------------------------------------------
    # Dense workspace
    # --------------------------------------------------------

    dense_folder = os.path.join(
        workspace_folder,
        "dense"
    )

    os.makedirs(
        dense_folder,
        exist_ok=True
    )

    # ========================================================
    # STEP 1 - IMAGE UNDISTORTER
    # ========================================================

    undistort_command = [
        COLMAP_EXE,
        "image_undistorter",

        "--image_path",
        image_folder,

        "--input_path",
        sparse_folder,

        "--output_path",
        dense_folder,

        "--output_type",
        "COLMAP"
    ]

    run_command(
        undistort_command
    )

    # ========================================================
    # STEP 2 - PATCH MATCH STEREO
    # ========================================================

    patch_match_command = [
        COLMAP_EXE,
        "patch_match_stereo",

        "--workspace_path",
        dense_folder,

        "--workspace_format",
        "COLMAP",

        "--PatchMatchStereo.geom_consistency",
        "true"
    ]

    run_command(
        patch_match_command
    )

    # ========================================================
    # STEP 3 - STEREO FUSION
    # ========================================================

    fused_path = os.path.join(
        dense_folder,
        "fused.ply"
    )

    fusion_command = [
        COLMAP_EXE,
        "stereo_fusion",

        "--workspace_path",
        dense_folder,

        "--workspace_format",
        "COLMAP",

        "--input_type",
        "geometric",

        "--output_path",
        fused_path
    ]

    run_command(
        fusion_command
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "dense_path": dense_folder,
        "fused_point_cloud": fused_path
    }