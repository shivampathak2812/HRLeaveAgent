import runpy


def main():
    # Run the HR Leave Agent chat loop defined in agent.py
    runpy.run_module("agent", run_name="__main__")


if __name__ == "__main__":
    main()