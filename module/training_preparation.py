import os
import json
import cv2
import random
import shutil
import zipfile


class Training_preparation_data_set:
    def __init__(self, Input_folder):
        
        def settings():
            
            self.input_folder = Input_folder
            
            self.image_files = [f for f in os.listdir(Input_folder) if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            self.label_files = [f for f in os.listdir(Input_folder) if f.lower().endswith(('.json'))]
                        
            
            self.OUTPUT_FOLDER_NAME = "train_data"
            
            self.OUTPUT_LABEL_FOLDER_TRAIN  = "train_data/labels/train"
            self.OUTPUT_LABEL_FOLDER_VAL    = "train_data/labels/val"
            self.OUTPUT_IMAGE_FOLDER_TRAIN  = "train_data/images/train"
            self.OUTPUT_IMAGE_FOLDER_VAL    = "train_data/images/val"
            
            self.TRAIN_VAL_PERCENTAGE = 0.2
            
            self.FOLDER_TO_ZIP = "train_data"
            self.ZIP_FILE_PATH = "dataset.zip"

        settings()
        
        
    def coordinates_to_yolo(self, coordinates, image_width, image_height, class_name):
        x_min, y_min, x_max, y_max = coordinates
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        return f"{class_name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    
    
    def create_training_folder_structure(self, base_dir):
        
        # Diretórios de imagens
        image_dir = os.path.join(base_dir, "images")
        train_image_dir = os.path.join(image_dir, "train")
        val_image_dir = os.path.join(image_dir, "val")

        # Diretórios de rótulos
        label_dir = os.path.join(base_dir, "labels")
        train_label_dir = os.path.join(label_dir, "train")
        val_label_dir = os.path.join(label_dir, "val")

        # Criar diretórios se eles não existirem
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(val_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)
        
        
    def write_file(self):
        
        for image_file, json_file in zip(self.image_files, self.label_files):
            image_path = os.path.join(self.input_folder, image_file)
            json_path  = os.path.join(self.input_folder, json_file)
            output_filename = os.path.splitext(os.path.basename(image_file))[0] + '.txt'
            
            imagem = cv2.imread(image_path)
            height, width = imagem.shape[:2]
            
            with open(json_path, 'r') as arquivo:
                data = json.load(arquivo)
            
            with open(os.path.join(self.OUTPUT_LABEL_FOLDER_TRAIN, output_filename), 'w') as file:
                if data:
                    for deteccao in data:
                        classe = deteccao['Classe']
                        x_min = deteccao['x_min']
                        y_min = deteccao['y_min']
                        x_max = deteccao['x_max']
                        y_max = deteccao['y_max']
                        
                        coordinates = (x_min, y_min, x_max, y_max)
                        
                        line_yolo = self.coordinates_to_yolo(coordinates, width, height, classe)
                        file.write(line_yolo)
                
                else: file.write("")
                
                
    def cop_images(self):
        for image_file in self.image_files:
            source_path =   os.path.join(self.input_folder, image_file)
            dest_path =     os.path.join(self.OUTPUT_IMAGE_FOLDER_TRAIN, image_file)
            shutil.copy2(source_path, dest_path)
    
        
    def organize_files(self, percentage):
            
        image_files_train = [f for f in os.listdir(self.OUTPUT_IMAGE_FOLDER_TRAIN)
                             if os.path.isfile(os.path.join(self.OUTPUT_IMAGE_FOLDER_TRAIN, f))]
        
        label_files_train = [f for f in os.listdir(self.OUTPUT_LABEL_FOLDER_TRAIN)
                             if os.path.isfile(os.path.join(self.OUTPUT_LABEL_FOLDER_TRAIN, f))]
        
        if len(image_files_train) == len(label_files_train):
        
            num_files_to_move = int(len(image_files_train) * (percentage))
            files_to_move_indices = random.sample(range(len(image_files_train)), num_files_to_move)

            for index in files_to_move_indices:
                image_file_to_move = image_files_train[index]
                label_file_to_move = label_files_train[index]
                
                source_image = os.path.join(self.OUTPUT_IMAGE_FOLDER_TRAIN, image_file_to_move)
                destination_image = os.path.join(self.OUTPUT_IMAGE_FOLDER_VAL, image_file_to_move)
                
                source_label = os.path.join(self.OUTPUT_LABEL_FOLDER_TRAIN, label_file_to_move)
                destination_label = os.path.join(self.OUTPUT_LABEL_FOLDER_VAL, label_file_to_move)
                
                shutil.move(source_image, destination_image)
                shutil.move(source_label, destination_label)
                
    def zip_folder(self, folder_path, zip_path):
        """
        Compacta uma pasta em um arquivo zip.

        :param folder_path: O caminho para a pasta que você deseja compactar.
        :param zip_path: O caminho para o arquivo zip de destino.
        """
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

    
    def run(self):
        self.create_training_folder_structure(self.OUTPUT_FOLDER_NAME)
        self.write_file()
        self.cop_images()
        self.organize_files(self.TRAIN_VAL_PERCENTAGE)
        self.zip_folder(self.FOLDER_TO_ZIP, self.ZIP_FILE_PATH)


if __name__ == "__main__":
    
    input_folder = r"C:\Users\Lucas Cordeiro\Desktop\teste5"
    
    training_preparation = Training_preparation_data_set(input_folder)
    training_preparation.run()
    