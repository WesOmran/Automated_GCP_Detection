import os
import laspy
import time

"""
    This method takes in parameters that define what
    classified points needed to be separated
    
    First Parameter (input_folder):
        User is going to prompted on the terminal to enter
        folder containing .LAS files that need to be separated
    
    Second Parameter (class_number):
        User will be prompted to select the class number
        Class 0: Created, never classified
        Class 1: Unclassified
        Class 2: Ground
        Class 3: Low Vegetation
        Class 4: Medium Vegetation
        Class 5: High Vegetation
        Class 6: Building
        Class 7: Low Point (Noise)
        Class 8: Model Key-Point (Mass Point)
        Class 9: Water
        and so on ...
        
    This method then automatically creates 2 folders inside the 
    input folder. Folders are created as follows:
        class_#: containing LAS only with specified class points
        no_class_#: containing LAS excluding specified class points
"""

def separate_points(input_folder, class_number):
    # Create subfolders
    output_folder = os.path.join(input_folder, "separation")
    specified_class_folder = os.path.join(output_folder, f"{class_description.get(class_number)}_class_only")
    other_folder = os.path.join(output_folder, f"no_{class_description.get(class_number)}_class")
    os.makedirs(specified_class_folder, exist_ok=True)
    os.makedirs(other_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".las"):
            each_file_start_time = time.time()
            
            input_path = os.path.join(input_folder, filename)
            print(f"\nProcessing: {input_path}")

            # Read the LAS file
            las = laspy.read(input_path)

            # Get ground (classification == 2) and non-ground points
            class_mask = las.classification == class_number
            other_mask = las.classification != class_number

            class_points = las.points[class_mask]
            non_class_points = las.points[other_mask]

            # Create new LAS files for ground and non-ground
            class_las = laspy.LasData(las.header)
            class_las.points = class_points

            non_class_las = laspy.LasData(las.header)
            non_class_las.points = non_class_points

            # Construct output file paths
            base_name = os.path.splitext(filename)[0]
            class_path = os.path.join(specified_class_folder, f"{base_name}_{class_description.get(class_number)}_class_only.las")
            non_class_path = os.path.join(other_folder, f"{base_name}_no_{class_description.get(class_number)}_class.las")

            # Write the output files
            class_las.write(class_path)
            non_class_las.write(non_class_path)
            
            each_file_end_time = time.time()
            
            each_file_execution_time = (each_file_end_time - each_file_start_time)
            
            print(f"\nSaved:\n{class_path}\n{non_class_path}")
            print(f"\nExecution time: {each_file_execution_time:.2f} seconds")
            
class_description = {
    0: "Created_never_classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low_Vegetation",
    4: "Medium_Vegetation",
    5: "High_Vegetation",
    6: "Building",
    7: "Low_Point_(noise)",
    8: "Model_Key-point",
    9: "Water",
    10: "Rail",
    11: "Road_Surface",
    12: "Reserved",
    13: "Wire_Guard",
    14: "Wire_Conductor",
    15: "Transmission_Tower",
    16: "Wire-structure_Connector",
    17: "Bridge_Deck",
    18: "High_Noise"
}

# Ask user for parameters
input_folder = input("Enter full path of folder containing las files:\n").strip()
for value in class_description:
    print(f"{value}:\t{class_description.get(value)}")
input_class = int(input("Enter class number to be filtered out: ").strip())

start_time = time.time()

separate_points(input_folder, input_class)

end_time = time.time()

execution_time = (end_time - start_time) / 60
print(f"\nTotal execution time: {execution_time:.2f} minutes")

