#  Biblioteca de preprocesamiento de datos de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Para las palabras raíz
from nltk.stem import PorterStemmer

# Crear una instancia para la clase PorterStemmer
stemmer = PorterStemmer()

# Importar la biblioteca json
import json
import pickle
import numpy as np

words=[] # Lista de raíces de palabras únicas en los datos
classes = [] # Lista de etiquetas únicas en los datos
pattern_word_tags_list = [] # Lista de pares de la forma (['palabras', 'de', 'las', 'oraciones'], 'etiquetas')

# Palabras que se ignorarán al crear el conjunto de datos
ignore_words = ['?', '!',',','.', "'s", "'m"]

# Abrir el archivo JSON y cargando sus datos
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# Creando una función para las palabras raíz
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:

     # Verificar si la palabra no es parte de palabras vacías:
        if word not in ignore_words:
            # Vuélvelas minúsculas
            w = word.lower()
            # Crea su raíz
            stem_word = stemmer.stem(w)
            # Añádelas a la lista stem_words
            stem_words.append(stem_word)
      

    return stem_words


'''
Lista ordenada de palabras raíz para nuestro conjunto de datos:
['all', 'ani', 'anyon', 'are', 'awesom', 'be', 'best', 'bluetooth', 'bye', 'camera', 'can', 'chat', 
'cool', 'could', 'digit', 'do', 'for', 'game', 'goodby', 'have', 'headphon', 'hello', 'help', 'hey', 
'hi', 'hola', 'how', 'is', 'later', 'latest', 'me', 'most', 'next', 'nice', 'phone', 'pleas', 'popular', 
'product', 'provid', 'see', 'sell', 'show', 'smartphon', 'tell', 'thank', 'that', 'the', 'there', 
'till', 'time', 'to', 'trend', 'video', 'what', 'which', 'you', 'your']
'''


# Creando una función para hacer el corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in data['intents']:

        # Agregar todos los patrones y las etiquetas a una lista
        for pattern in intent['patterns']:  

            # Tokenizar el patrón
            pattern_words = nltk.word_tokenize(pattern)

            # Agregar las palabras tokenizadas a la lista "words"
            words.extend(pattern_words)
            
            # Agregar la lista de palabras tokenizadas junto con la etiqueta a la lista pattern_word_tags_list
            pattern_word_tags_list.append((pattern_words, intent['tag']))
            
            
        # Agregar todas las etiquetas a la lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

            
    stem_words = get_stem_words(words, ignore_words) 

    # Remover palabras duplicadas de stem_words

    # Ordenar las listas stem_words y classes

    
    # Imprimir stem_words
    print('Lista stem_words: ' , stem_words)

    return stem_words, classes, pattern_word_tags_list


# Conjunto de datos de entrenamiento: 
# Texto de entrada ----> como bolsa de palabras
# Etiquetas -----------> como etiqueta

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    
    bag = []
    for word_tags in pattern_word_tags_list:
        # Ejemplo: word_tags = (['hi', 'there'], 'greetings']

        pattern_words = word_tags[0] # ['Hi' , 'There]
        bag_of_words = []

        # Patrones de palabras para stemming antes de crear la bolsa de palabras
        stemmed_pattern_words = get_stem_words(pattern_words, ignore_words)

        # Codificación de datos de entrada
        for word in stem_words:
            if word in stemmed_pattern_words:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        
        bag.append(bag_of_words)
    
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    
    labels = []

    for word_tags in pattern_word_tags_list:

        # Comenzar con la lista de ceros
        labels_encoding = list([0]*len(classes))  

        # Ejemplo: word_tags = (['hi', 'there'], 'greetings']

        tag = word_tags[1]   # 'greetings'

        tag_index = classes.index(tag)

        # Códificación de etiquetas
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)
        
    return np.array(labels)

def preprocess_train_data():
  
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Convertir las palabras raíz y las clases a formato de archivo pickel de Python
    with open('words.pkl', 'wb') as words_file:
        pickle.dump(stem_words, words_file)

    with open('classes.pkl', 'wb') as classes_file:
        pickle.dump(tag_classes, classes_file)

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

bow_data  , label_data = preprocess_train_data()

# Después de completar el código, remueve el comentario de las declaraciones de impresión
print("Primera codificación de la bolsa de palabras: " , bow_data[0])
print("Primera codificación de las etiquetas: " , label_data[0])

