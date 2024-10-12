import csv

def write_csv(data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

        # Define headers: frame number + joint coordinates (e.g., x1, y1, z1, x2, y2, z2, ...)
        headers = ['frame'] + [f'x{i//3+1}, y{i//3+1}, z{i//3+1}' for i in range(51)]  # 17 joints * 3 coordinates
        writer.writerow(headers)

        # Write the data
        for i, frame in enumerate(data):
            writer.writerow([i] + frame)
