import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"

    if cmd == "train":
        from train import train
        train()

    elif cmd == "evaluate":
        from infer import evaluate
        evaluate()

    elif cmd == "infer":
        index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        from infer import infer
        infer(indent=index)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
