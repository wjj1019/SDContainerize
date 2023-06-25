from pdf2image import convert_from_path
import os
import cv2
import pytesseract
import re
from PIL import Image
from pytesseract import Output
import numpy as np
import json
import pandas as pd

# The function that processes the image
def process_image(image_path):
    
    #Field Names
    field_names = ['Employment Income', 'Income Tax Deducted', 'CPPs Contribution', 'EI Earnings', 'QPP Contributions', 'Pensionable Earnings', 'EI Premiums', 'Union Dues',
              'RPP Contribution', 'Charitable Donations', 'Pension Adjustment', 'RPP or DPSP Registration Number', 'PPIP Premiums', 'PPIP Earnings','Employers Name', 'Employment account number', 'SIN', 'Year',
              'Province', 'Employment Code', 'Employee Name', 'Address', 'CPP/QPP', 'EI', 'PPIP', 'Box-Case1', 'Amount1', 'Box-Case2', 'Amount2', 'Box-Case3', 'Amount3']

    # Trained values from model     
    ratios = [(-10.2, 5.5, 8, 1.9),(-0.8, 5.5, 7, 1.9),         #Box 14, 22
            (-8.7, 9.1, 6.5, 1.9), (-0.3, 9.1, 6.5, 1.9),    #Box 16, 24 
            (-8.7, 12.5, 6.5, 1.9), (-0.3, 12.5, 6.5, 1.9),   #Box 17, 26
            (-8.7, 16, 6.5, 1.9), (-0.3, 16, 6.5, 1.9),       #Box 18, 44
            (-8.7, 19.5, 6.5, 1.9), (-0.3, 19.5, 6.5, 1.9),   #Box 20, 46
            (-8.7, 23.2, 6.5, 1.9), (-0.3, 23.2, 6.5, 1.9),   #Box 52, 50
            (-8.7, 26.7, 6.5, 1.9), (-0.3, 26.7, 6.5, 1.9),   #Box 55, 56
            (-28, 1, 13.5, 6),                                #Employer Name
            (-27, 8.2 , 12.5, 1.5),                           #EIN
            (-27, 11.6, 7.7, 1.9),                            #SIN
            (-12.3, 1.4, 3.5, 1.9) , (-12.5, 9, 1.9, 1.9) , (-12.5, 12.4, 1.9, 1.9), #Year, 10, 29 
            (-27.4, 17.5, 16.5, 1.9), (-27.5, 19.5, 17, 8), #Name and Address
            (-17.8, 11.6, 1, 1.9), (-16.4, 11.6, 1, 1.9), (-15.1, 11.6, 1, 1.9), #Little Boxes
            (-24, 30.3, 2 ,1.9), (-21.5, 30.3, 7 ,1.9), (-13.8, 30.3, 2 ,1.9), (-11.2, 30.3, 7 ,1.9), (-3.2, 30.3, 2 ,1.9), (-1, 30.3, 7.3 ,1.9)]


    image = cv2.imread(image_path)

    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    coords = []

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if data['text'][i] == "T4":
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            coords.append((x, y, w, h))

        elif data['text'][i] == "T":
            if i < n_boxes - 1 and data['text'][i+1] == "4":
                x = min(data['left'][i], data['left'][i+1])
                y = min(data['top'][i], data['top'][i+1])
                w = max(data['left'][i] + data['width'][i], data['left'][i+1] + data['width'][i+1]) - x
                h = max(data['top'][i] + data['height'][i], data['top'][i+1] + data['height'][i+1]) - y
                coords.append((x, y, w, h))

    x, y, w, h = sorted(coords, key=lambda coord: coord[1])[0]

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    extracted_text = {}
    for field_name, (relative_x_ratio, relative_y_ratio, relative_w_ratio, relative_h_ratio) in zip(field_names, ratios):
        relative_x = int(relative_x_ratio * w)
        relative_y = int(relative_y_ratio * h)
        relative_w = int(relative_w_ratio * w)
        relative_h = int(relative_h_ratio * h)
        cv2.rectangle(image, (x + relative_x, y + relative_y), (x + relative_x + relative_w, y + relative_y + relative_h), (0, 0, 255), 2)

        roi = image[y + relative_y:y + relative_y + relative_h, x + relative_x:x + relative_x + relative_w]
        text = pytesseract.image_to_string(roi).strip()

        extracted_text[field_name] = text
    
    def clean_monetary(s):
        # Remove any characters that are not digits or a dot
        s = re.sub(r"[^\d.]", "", s)
        # Remove duplicate dots, if any
        s = re.sub(r"\.{2,}", ".", s)
        return s

    def clean_alpha_numeric(s):
        # Replace newline characters with a space
        s = re.sub(r"\n", " ", s)
        # Remove any characters that are not alphanumeric or a space
        s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
        return s.strip()

    def clean_address(s):
        # Replace newline characters with a comma and a space
        s = re.sub(r"\n", ", ", s)
        return s.strip()

    def clean_numeric(s):
        # Remove any characters that are not digits
        s = re.sub(r"\D", "", s)
        return s

    def clean_alpha(s):
        # Remove any characters that are not letters
        s = re.sub(r"[^a-zA-Z]", "", s)
        return s

    def clean_year(s):
        # Only keep the last four digits (assuming it's the most recent year)
        s = re.sub(r"\D", "", s)
        return s[-4:]

    cleaning_functions = {
        'Employment Income': clean_monetary,
        'Income Tax Deducted': clean_monetary,
        'CPPs Contribution': clean_monetary,
        'EI Earnings': clean_monetary,
        'QPP Contributions': clean_monetary,
        'Pensionable Earnings': clean_monetary,
        'EI Premiums': clean_monetary,
        'Union Dues': clean_monetary,
        'RPP Contribution': clean_monetary,
        'Charitable Donations': clean_monetary,
        'Pension Adjustment': clean_monetary,
        'RPP or DPSP Registration Number': clean_alpha_numeric,
        'PPIP Premiums': clean_monetary,
        'Employers Name': clean_alpha_numeric,
        'EIN': clean_alpha_numeric,
        'SIN': clean_alpha_numeric,
        'Year': clean_year,
        'Province': clean_alpha,
        'Employment Code': clean_alpha,
        'Employee Name': clean_alpha_numeric,
        'Address': clean_address,
        'CPP/QPP': clean_numeric,
        'EI': clean_numeric,
        'PPIP': clean_numeric,
        'Box-Case1': clean_alpha_numeric,
        'Amount1': clean_monetary,
        'Box-Case2': clean_alpha_numeric,
        'Amount2': clean_monetary,
    }

    # Now apply each cleaning function to the corresponding field in your data dictionary
    for key, value in extracted_text.items():
        if key in cleaning_functions:
            extracted_text[key] = cleaning_functions[key](value)


    json_data = json.dumps(extracted_text)

    return json_data

    
# Main function that determines the file type and performs appropriate actions
def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)

    results = []

    if file_extension.lower() in ['.png', '.jpg', '.jpeg']:
        result = process_image(file_path)
        results.append(result)

    elif file_extension.lower() == '.pdf':
        # Convert the pdf to a list of images
        images = convert_from_path(file_path, first_page=0, last_page=1)

        # Process each image
        for image in images:
            # Save each image as a temp file and get its path
            temp_image_path = "temp_image.png"
            image.save(temp_image_path, "PNG")

            # Process the image and append the result
            result = process_image(temp_image_path)
            results.append(result)

            # Remove the temp file
            os.remove(temp_image_path)

    else:
        print("Unsupported file type")

    # Return the list of results
    return results

def convert_pdf_to_img(pdf_path):
    return convert_from_path(pdf_path)

def ocr_image(image):
    return pytesseract.image_to_string(image)