import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

# --- Configurações --- #
DATA_DIRS = ['captcha_example_images', '/home/guilherme/Documentos/GitHub/automate/captcha_cnn/imagens'] # Pasta com as imagens rotuladas
MODEL_FILENAME = 'captcha_com_model.keras' # Nome do arquivo do novo modelo
CONFIG_FILENAME = 'model_config.json' # Nome do arquivo de configuração

# Dimensões das imagens (devem ser as mesmas usadas na predição)
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 120

# Parâmetros de Treinamento
EPOCHS = 100 # Aumentado para dar mais tempo com EarlyStopping
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2 # Usar 20% dos dados para validação

def load_and_preprocess_data():
    print("Carregando e pré-processando dados de múltiplos diretórios...")
    all_images = []
    all_labels = []

    for data_dir in DATA_DIRS:
        print(f"Processando diretório: {data_dir}")
        if not os.path.exists(data_dir):
            print(f"Aviso: Diretório {data_dir} não encontrado. Pulando.")
            continue

        # Lógica para o diretório 'captcha_example_images' que usa labels.csv
        if 'captcha_example_images' in data_dir:
            labels_path = os.path.join(data_dir, 'labels.csv')
            if not os.path.exists(labels_path):
                print(f"Aviso: {labels_path} não encontrado. Pulando este diretório.")
                continue
            labels_df = pd.read_csv(labels_path)
            for _, row in labels_df.iterrows():
                filename = row['filename']
                label = row['label']
                img_path = os.path.join(data_dir, filename)
                if os.path.exists(img_path):
                    all_labels.append(label)
                    all_images.append(img_path)
        else:
            # Lógica para outros diretórios onde o nome do arquivo é o rótulo
            for filename in os.listdir(data_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = os.path.splitext(filename)[0]
                    img_path = os.path.join(data_dir, filename)
                    all_labels.append(label)
                    all_images.append(img_path)

    images_processed = []
    labels_processed = []
    for i, img_path in enumerate(all_images):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Aviso: Não foi possível ler a imagem {img_path}. Pulando.")
                continue
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            images_processed.append(img)
            labels_processed.append(all_labels[i])
        except Exception as e:
            print(f"Erro ao processar a imagem {img_path}: {e}")

    if not images_processed or not labels_processed:
        print("Erro: Nenhuma imagem ou rótulo foi carregado. Verifique os diretórios de dados.")
        sys.exit(1)

    print(f"Total de {len(images_processed)} imagens carregadas.")
    return np.array(images_processed), labels_processed

def build_model(img_height, img_width, max_length, num_chars):
    """Constrói o modelo CNN com data augmentation."""
    input_layer = layers.Input(shape=(img_height, img_width, 1), name='input')
    
    # Camada de aumento de dados para gerar mais exemplos de treinamento
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.05, fill_mode='nearest'),
        layers.RandomZoom(0.1, fill_mode='nearest'),
        layers.RandomContrast(0.1)
    ], name='data_augmentation')
    
    x = data_augmentation(input_layer)

    # Blocos Convolucionais
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Camadas densas
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Camadas de saída, uma para cada caractere do CAPTCHA
    outputs = [
        layers.Dense(num_chars, activation='softmax', name=f'char_{i}')(x)
        for i in range(max_length)
    ]

    model = models.Model(inputs=input_layer, outputs=outputs, name='captcha_solver')
    
    # Para um modelo com múltiplas saídas, precisamos especificar a perda e a métrica para cada saída.
    # Criamos dicionários que mapeiam o nome de cada saída para a função desejada.
    losses = {f'char_{i}': 'categorical_crossentropy' for i in range(max_length)}
    metrics = {f'char_{i}': 'accuracy' for i in range(max_length)}

    # Compila o modelo com os dicionários de perda e métricas
    model.compile(optimizer='adam', loss=losses, metrics=metrics)
    return model

if __name__ == '__main__':
    print("--- Iniciando script de treinamento ---")
    # 1. Carregar dados
    images, labels = load_and_preprocess_data()
    if not images.size > 0:
        print("ERRO: Nenhum dado foi carregado. Saindo.")
        exit()
    print("--- Passo 1/7: Dados carregados ---")

    # 2. Determinar parâmetros a partir dos dados
    max_length = max(len(label) for label in labels)
    all_chars = sorted(list(set(''.join(labels))))
    num_chars = len(all_chars)
    char_to_num = {char: i for i, char in enumerate(all_chars)}
    print(f"Comprimento máximo do CAPTCHA: {max_length}")
    print(f"Caracteres encontrados ({num_chars}): {''.join(all_chars)}")
    print("--- Passo 2/7: Parâmetros determinados ---")

    # 3. Codificar rótulos
    encoded_labels = np.zeros((len(labels), max_length, num_chars), dtype='uint8')
    for i, label in enumerate(labels):
        for j, char in enumerate(label):
            # Garante que o caractere existe no nosso dicionário
            if char in char_to_num:
                encoded_labels[i, j, char_to_num[char]] = 1
    y = [encoded_labels[:, i, :] for i in range(max_length)]
    print("--- Passo 3/7: Rótulos codificados ---")

    # 4. Dividir dados em treino e validação
    split_result = train_test_split(images, *y, test_size=VALIDATION_SPLIT, random_state=42)
    X_train, X_val = split_result[0], split_result[1]
    y_train_list = [split_result[i] for i in range(2, len(split_result), 2)]
    y_val_list = [split_result[i] for i in range(3, len(split_result), 2)]
    y_train = {f'char_{i}': y_train_list[i] for i in range(max_length)}
    y_val = {f'char_{i}': y_val_list[i] for i in range(max_length)}
    print("--- Passo 4/7: Dados divididos ---")

    # 5. Construir o modelo
    model = build_model(IMAGE_HEIGHT, IMAGE_WIDTH, max_length, num_chars)
    model.summary()
    print("--- Passo 5/7: Modelo construído ---")

    # 6. Callbacks para otimizar o treinamento
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]
    print("--- Passo 6/7: Callbacks definidos ---")

    # 7. Treinar o modelo
    print("\n--- Passo 7/7: Iniciando o treinamento do modelo... ---")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Salvar o modelo e a configuração
    model.save(MODEL_FILENAME)
    config_data = {
        'max_length': max_length,
        'characters': ''.join(all_chars)
    }
    with open(CONFIG_FILENAME, 'w') as f:
        json.dump(config_data, f)

    print(f"\nTreinamento concluído! Modelo salvo como '{MODEL_FILENAME}' e configuração como '{CONFIG_FILENAME}'.")
