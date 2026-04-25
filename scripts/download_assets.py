from midway_project.models import download_required_models


def main() -> None:
    download_required_models()
    print("Model downloads completed.")


if __name__ == "__main__":
    main()
