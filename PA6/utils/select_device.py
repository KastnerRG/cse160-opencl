import subprocess
import re
import json

def get_device_type(platform_id=0, device_id=0):
    output = subprocess.check_output(["clinfo", "-d", f"{platform_id}:{device_id}", "--prop", "cl_device_type"], text=True)
    match = re.search(r".+CL_DEVICE_TYPE_(\w+)", output)
    if match:
        device_type = match.group(1)
        return device_type
    else:
        raise ValueError("Could not determine device type from clinfo output.")

def print_selected_device_info(platform_id, device_id, devices):
    platform_info = devices[platform_id]
    device_info = platform_info["devices"][device_id]
    print(f"Selected Device: Platform {platform_id} - {platform_info['name']}, Device {device_id} - {device_info['name']} (Type: {device_info['type']})")

def select_device(selected_plat_id = None, selected_device_id = None, selected_device_type = "GPU", return_all=False):
    if selected_device_type not in [None, "CPU", "GPU"]:
        raise ValueError("selected_device_type must be one of None, 'CPU', or 'GPU'")

    output = subprocess.check_output(["clinfo", "-l"], text=True)

    devices = {}
    platform_id = 0
    device_count = 0

    # Get all Devices
    for line in output.splitlines():
        match = re.match(r"Platform #(\d+):([\w ()-]+)", line)
        if match:
            platform_id = int(match.group(1))
            platform_name = match.group(2)
            devices[platform_id] = platform_name
            devices[platform_id] = {
                "name": platform_name,
                "devices": {}
            }
            continue


        match = re.match(r".+Device #(\d+):([\w ()-]+)", line)
        if match:
            device_id = int(match.group(1))
            device_name = match.group(2)
            devices[int(platform_id)]["devices"][device_id] = {
                "name": device_name,
                "pytorch_ocl_id": f"ocl:{device_count}",
                "type": get_device_type(platform_id=platform_id, device_id=device_id)
            }
            device_count += 1
    
    if return_all:
        return devices

    # Filter Devices on User Selection
    if selected_plat_id is not None and selected_device_id is not None:
        if selected_plat_id in devices and selected_device_id in devices[selected_plat_id]["devices"]:
            print_selected_device_info(selected_plat_id, selected_device_id, devices)
            return devices[selected_plat_id]["devices"][selected_device_id]["pytorch_ocl_id"]
        else:
            print("Selected platform or device ID not found. Falling back to automatic selection.")

    # Fallback
    selected_plat_id = None
    selected_device_id = None
    for platform_id, platform_info in devices.items():
        for device_id, device_info in platform_info["devices"].items():
            if device_info["type"] == selected_device_type or selected_device_id is None:
                if selected_plat_id is not None \
                    and selected_device_id is not None \
                    and devices[selected_plat_id]["name"] != "Portable Computing Language":
                        # Do not select pocl device if a non-pocl device was found
                        continue
                
                selected_device_id = device_id
                selected_plat_id = platform_id
    
    print_selected_device_info(selected_plat_id, selected_device_id, devices)
    return devices[selected_plat_id]["devices"][selected_device_id]["pytorch_ocl_id"]

if __name__ == "__main__":
    devices = select_device(selected_plat_id=1, selected_device_id=0, selected_device_type="GPU", return_all=False)

    print(devices)
