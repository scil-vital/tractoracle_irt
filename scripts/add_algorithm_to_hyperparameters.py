import json
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Add or check 'algorithm' field in a hyperparameters.json file."
    )
    parser.add_argument("json_file", help="Path to hyperparameters.json")
    parser.add_argument("algorithm", help="Algorithm value to set")
    args = parser.parse_args()

    json_path = Path(args.json_file)

    if not json_path.exists():
        print(f"Error: File {json_path} does not exist.")
        sys.exit(1)

    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON. {e}")
            sys.exit(1)

    algo_value = args.algorithm

    if "algorithm" in data:
        if data["algorithm"] == algo_value:
            print("Field 'algorithm' already set to the same value. Skipping.")
        else:
            print(
                f"Warning: Field 'algorithm' exists with value '{data['algorithm']}'. "
                f"Requested value is '{algo_value}'. No changes made."
            )
    else:
        data["algorithm"] = algo_value
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Field 'algorithm' added with value '{algo_value}'.")

if __name__ == "__main__":
    main()