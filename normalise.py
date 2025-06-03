import os
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, save_img

# Target number of images per class
TARGET_COUNT = 300

# Your dataset path
root_folder = '/Users/emmanuelgeorgep/Documents/College/HonoursProject/balanced_test'

# Valid image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def get_image_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(image_extensions) and not f.startswith('aug_')]

# Process each class folder
for class_name in os.listdir(root_folder):
    class_folder = os.path.join(root_folder, class_name)
    if not os.path.isdir(class_folder):
        continue

    images = get_image_files(class_folder)
    current_count = len(images)
    print(f"\n📁 {class_name}: {current_count} images")

    # UNDERSAMPLE (delete random files)
    if current_count > TARGET_COUNT:
        to_delete = current_count - TARGET_COUNT
        print(f"⚠️  Deleting {to_delete} extra images...")
        delete_images = random.sample(images, to_delete)
        for file in delete_images:
            os.remove(os.path.join(class_folder, file))

    # OVERSAMPLE (augment)
    elif current_count < TARGET_COUNT:
        to_generate = TARGET_COUNT - current_count
        print(f"➕ Generating {to_generate} new images with augmentation...")
        i = 0
        while i < to_generate:
            src_img = random.choice(images)
            src_path = os.path.join(class_folder, src_img)
            img = load_img(src_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            for batch in datagen.flow(x, batch_size=1):
                new_name = f"aug_{i}.jpg"
                save_img(os.path.join(class_folder, new_name), array_to_img(batch[0]))
                i += 1
                if i >= to_generate:
                    break

    else:
        print("✅ Already has 1000 images. No changes needed.")

print("\n🎯 All class folders now contain exactly 1300 images.")