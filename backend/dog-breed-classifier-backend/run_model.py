from extract_bottleneck_features import extract_Resnet50
import tensorflow as tf
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
from io import BytesIO
from PIL import Image
import base64

dog_names=['Affenpinscher',
 'Afghan_hound',
 'Airedale_terrier',
 'Akita',
 'Alaskan_malamute',
 'American_eskimo_dog',
 'American_foxhound',
 'American_staffordshire_terrier',
 'American_water_spaniel',
 'Anatolian_shepherd_dog',
 'Australian_cattle_dog',
 'Australian_shepherd',
 'Australian_terrier',
 'Basenji',
 'Basset_hound',
 'Beagle',
 'Bearded_collie',
 'Beauceron',
 'Bedlington_terrier',
 'Belgian_malinois',
 'Belgian_sheepdog',
 'Belgian_tervuren',
 'Bernese_mountain_dog',
 'Bichon_frise',
 'Black_and_tan_coonhound',
 'Black_russian_terrier',
 'Bloodhound',
 'Bluetick_coonhound',
 'Border_collie',
 'Border_terrier',
 'Borzoi',
 'Boston_terrier',
 'Bouvier_des_flandres',
 'Boxer',
 'Boykin_spaniel',
 'Briard',
 'Brittany',
 'Brussels_griffon',
 'Bull_terrier',
 'Bulldog',
 'Bullmastiff',
 'Cairn_terrier',
 'Canaan_dog',
 'Cane_corso',
 'Cardigan_welsh_corgi',
 'Cavalier_king_charles_spaniel',
 'Chesapeake_bay_retriever',
 'Chihuahua',
 'Chinese_crested',
 'Chinese_shar-pei',
 'Chow_chow',
 'Clumber_spaniel',
 'Cocker_spaniel',
 'Collie',
 'Curly-coated_retriever',
 'Dachshund',
 'Dalmatian',
 'Dandie_dinmont_terrier',
 'Doberman_pinscher',
 'Dogue_de_bordeaux',
 'English_cocker_spaniel',
 'English_setter',
 'English_springer_spaniel',
 'English_toy_spaniel',
 'Entlebucher_mountain_dog',
 'Field_spaniel',
 'Finnish_spitz',
 'Flat-coated_retriever',
 'French_bulldog',
 'German_pinscher',
 'German_shepherd_dog',
 'German_shorthaired_pointer',
 'German_wirehaired_pointer',
 'Giant_schnauzer',
 'Glen_of_imaal_terrier',
 'Golden_retriever',
 'Gordon_setter',
 'Great_dane',
 'Great_pyrenees',
 'Greater_swiss_mountain_dog',
 'Greyhound',
 'Havanese',
 'Ibizan_hound',
 'Icelandic_sheepdog',
 'Irish_red_and_white_setter',
 'Irish_setter',
 'Irish_terrier',
 'Irish_water_spaniel',
 'Irish_wolfhound',
 'Italian_greyhound',
 'Japanese_chin',
 'Keeshond',
 'Kerry_blue_terrier',
 'Komondor',
 'Kuvasz',
 'Labrador_retriever',
 'Lakeland_terrier',
 'Leonberger',
 'Lhasa_apso',
 'Lowchen',
 'Maltese',
 'Manchester_terrier',
 'Mastiff',
 'Miniature_schnauzer',
 'Neapolitan_mastiff',
 'Newfoundland',
 'Norfolk_terrier',
 'Norwegian_buhund',
 'Norwegian_elkhound',
 'Norwegian_lundehund',
 'Norwich_terrier',
 'Nova_scotia_duck_tolling_retriever',
 'Old_english_sheepdog',
 'Otterhound',
 'Papillon',
 'Parson_russell_terrier',
 'Pekingese',
 'Pembroke_welsh_corgi',
 'Petit_basset_griffon_vendeen',
 'Pharaoh_hound',
 'Plott',
 'Pointer',
 'Pomeranian',
 'Poodle',
 'Portuguese_water_dog',
 'Saint_bernard',
 'Silky_terrier',
 'Smooth_fox_terrier',
 'Tibetan_mastiff',
 'Welsh_springer_spaniel',
 'Wirehaired_pointing_griffon',
 'Xoloitzcuintli',
 'Yorkshire_terrier']

def init_model():
    model=tf.keras.models.load_model('models/resnet50')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.load_weights('saved_models/weights.best.ResNet50.hdf5')

    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    test_DogResnet50 = bottleneck_features['test']

    Resnet50_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_DogResnet50[:1]]

    return model

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def bytes_to_tensor(img_bytes):
    img_bytes=img_bytes.split(",")[1]
    img_bytes=img_bytes.encode('utf-8')
    img_bytes=base64.b64decode(img_bytes)
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    x = np.asarray(img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    return x

def dog_breed(img_path,model):
    bottleneck_feature = extract_Resnet50(bytes_to_tensor(img_path))
    predicted_vector = model.predict(bottleneck_feature)
    top5_idx=np.argsort(predicted_vector[0])[::-1][:5]
    return[{'name':dog_names[idx],'probability':float(predicted_vector[0][idx])} for idx in top5_idx]

def dog_breed_detector(img_path,model):
    breed = dog_breed(img_path,model) 
    return breed
    

