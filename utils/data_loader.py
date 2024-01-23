import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    base_dir = "./"
    classes_folder_names = []
    def __init__(self, base_dir, classes_folder_names):
        self.base_dir = base_dir
        self.classes_folder_names = classes_folder_names

    def get_dataframes(self, train_ratio =0.8, test_ratio=0.1, proportional_testing_set=False):
        if(train_ratio + test_ratio > 1):
            raise ValueError("The train and test ratios must not be greater than 1.")

        data_frames = {}
        for class_name, i in self.classes_folder_names:
            data_frames[class_name] = pd.DataFrame(columns=["filename", "class"])
            class_dir = os.path.join(self.base_dir, class_name)
            x = 0
            for filename in os.listdir(class_dir):
                if x >= 3311:
                    break
                else:
                    x+=1

                data_frames[class_name] = pd.concat([data_frames[class_name], pd.DataFrame({"filename": [os.path.join(class_dir, filename)], "class": [class_name]})], ignore_index=True)

        train_data_frame = pd.DataFrame(columns=["filename", "class"])
        test_data_frame = pd.DataFrame(columns=["filename", "class"])
        val_data_frame = pd.DataFrame(columns=["filename", "class"])

        for class_name, i in self.classes_folder_names:
            class_data_frame = data_frames[class_name].reset_index(drop=True)
            train_temp = class_data_frame.sample(frac=train_ratio)
            train_data_frame = pd.concat([train_data_frame, train_temp], ignore_index=True)
            remaining = class_data_frame.drop(train_temp.index)
            test_temp = remaining.sample(frac=test_ratio/(1-train_ratio))
            test_data_frame = pd.concat([test_data_frame, test_temp], ignore_index=True)
            val_data_frame = pd.concat([val_data_frame, remaining.drop(test_temp.index)], ignore_index=True)

        if proportional_testing_set:
            min_test_samples = min(test_data_frame['class'].value_counts())
            test_data_frame = test_data_frame.groupby('class').apply(lambda x: x.sample(min_test_samples)).reset_index(drop=True)

        train_data_frame = train_data_frame.sample(frac=1).reset_index(drop=True)
        test_data_frame = test_data_frame.sample(frac=1).reset_index(drop=True)
        val_data_frame = val_data_frame.sample(frac=1).reset_index(drop=True)

        return train_data_frame, test_data_frame, val_data_frame

    def get_generators(self, train_data_frame, test_data_frame, val_data_frame, batch_size=32, target_size=(224, 224), data_augmentation=False):
            if not isinstance(train_data_frame, pd.DataFrame) or not isinstance(test_data_frame, pd.DataFrame) or not isinstance(val_data_frame, pd.DataFrame):
                raise ValueError("train_data, test_data, and val_data must be pandas DataFrames.")
            
            if train_data_frame.shape[1] != 2 or test_data_frame.shape[1] != 2 or val_data_frame.shape[1] != 2:
                raise ValueError("train_data, test_data, and val_data must have exactly two columns.")

            if(data_augmentation):
                train_datagen = ImageDataGenerator(rescale=1./255,
                                                    rotation_range=40,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2,
                                                    horizontal_flip=True)
                test_datagen = ImageDataGenerator(rescale=1./255, 
                                                    rotation_range=40,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2,
                                                    horizontal_flip=True)
                val_datagen = ImageDataGenerator(rescale=1./255,
                                                    rotation_range=40,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2,
                                                    horizontal_flip=True)
            else:
                train_datagen = ImageDataGenerator(rescale=1./255)
                test_datagen = ImageDataGenerator(rescale=1./255)
                val_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_dataframe(train_data_frame, ".", target_size=target_size, batch_size=batch_size, class_mode='categorical')
            test_generator = test_datagen.flow_from_dataframe(test_data_frame, ".", target_size=target_size, batch_size=batch_size, class_mode='categorical')
            val_generator = val_datagen.flow_from_dataframe(val_data_frame, ".", target_size=target_size, batch_size=batch_size, class_mode='categorical')

            return train_generator, test_generator, val_generator
