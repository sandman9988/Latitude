-
def update_file(filepath: str, param_name: str, value: float, instrument: str = "XAUUSD_M5_default"):
    """Update a parameter in one file with CRC32 validation."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Check if instrument exists
    if instrument not in data["data"]["instruments"]:
        print(f"✗ {filepath}: {instrument} not found")
        return False

    # Check if parameter exists
    params = data["data"]["instruments"][instrument]["params"]
    if param_name not in params:
        print(f"✗ {filepath}: {param_name} not found")
        return False

    # Get old value
    old_value = params[param_name]["value"]

    # Update value
    params[param_name]["value"] = value

    # Update timestamp
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Recalculate CRC32
    data["crc32"] = calculate_crc32(data["data"])

    # Save
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ {filepath}")
    print(f"  {param_name}: {old_value} → {value}")
    print(f"  CRC32: {data['crc32']}")
    return True


def list_parameters(filepath: str = "data/learned_parameters.json", instrument: str = "XAUUSD_M5_default"):
    """List all parameters in the file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if instrument not in data["data"]["instruments"]:
        print(f"Instrument {instrument} not found")
        return

    params = data["data"]["instruments"][instrument]["params"]

    print(f"\nParameters in {instrument}:")
    print("-" * 80)

    # Group by category
    categories = {"Harvester": [], "Risk": [], "Other": []}

    for name, param_data in sorted(params.items()):
        value = param_data["value"]
        if name.startswith("harvester_"):
            categories["Harvester"].append((name, value))
        elif any(x in name for x in ["risk", "var", "drawdown"]):
            categories["Risk"].append((name, value))
        else:
            categories["Other"].append((name, value))

    for category, items in categories.items():
        if items:
            print(f"\n{category}:")
            for name, value in items:
                print(f"  {name:40s} = {value}")


def main():
    parser = argparse.ArgumentParser(description="Update learned parameters with CRC32 validation")
    parser.add_argument("--param", help="Parameter name to update")
    parser.add_argument("--value", type=float, help="New value")
    parser.add_argument("--instrument", default="XAUUSD_M5_default", help="Instrument key")
    parser.add_argument("--all-files", action="store_true", help="Update all files including backups")
    parser.add_argument("--list", action="store_true", help="List all parameters")

    args = parser.parse_args()

    if args.list:
        list_parameters()
        return

    if not args.param or args.value is None:
        parser.print_help()
        return

    if args.all_files:
        # Update all files
        files = sorted(glob.glob("data/learned_parameters.json*"))
        print(f"Updating {len(files)} files...")
        print()

        success_count = 0
        for filepath in files:
            if update_file(filepath, args.param, args.value, args.instrument):
                success_count += 1
            print()

        print(f"Updated {success_count}/{len(files)} files successfully")
    else:
        # Update main file only
        update_file("data/learned_parameters.json", args.param, args.value, args.instrument)


if __name__ == "__main__":
    main()
