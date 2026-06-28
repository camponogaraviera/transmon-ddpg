from pathlib import Path
import imageio.v2 as imageio


def create_gif(
    frames_dir: str | Path,
    output_dir: str | Path,
    gif_name: str,
) -> None:
    """Create a GIF from PNG images in a directory.

    Args:
        frames_dir: Directory containing PNG frames.
        output_dir: Directory where the GIF will be saved.
        gif_name: Filename for the output GIF.

    Returns:
        None.
    """

    print(f"Creating {gif_name}...")

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    gif_name = Path(gif_name).with_suffix(".gif").name
    output_path = output_dir / gif_name

    files = sorted(frames_dir.glob("*.png"), key=lambda p: p.stem)
    if not files:
        raise ValueError(f"No PNG frames found in {frames_dir}")

    with imageio.get_writer(output_path, mode="I", duration=0.2, loop=0) as writer:
        for file in files:
            frame = imageio.imread(file)
            writer.append_data(frame)
