import argparse

from midway_project.data import prepare_coco_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the local COCO 2017 midway subset.")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10623)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--overwrite-manifest", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_coco_subset(
        subset_size=args.subset_size,
        seed=args.seed,
        image_size=args.image_size,
        overwrite_manifest=args.overwrite_manifest,
        max_workers=args.max_workers,
    )
    print(f"Prepared {len(manifest)} samples.")


if __name__ == "__main__":
    main()
