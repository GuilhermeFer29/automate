from __future__ import annotations
import os, re, glob, random, string, argparse
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

# Configurações
IMG_HEIGHT = 28
IMG_WIDTH = 28
CHARS = string.ascii_uppercase + string.digits  # 36 classes
DEFAULT_DATA_DIR = Path(__file__).with_name('imagens')
MIN_CHAR_AREA = 20
BATCH_SIZE = 128  # Tamanho de lote equilibrado para evitar problemas de memória
VAL_SPLIT = 0.2  # Reduzido para 20% de dados de validação

def find_label(file_name: str) -> str | None:
    """Extrai o rótulo do nome do arquivo"""
    name = Path(file_name).stem
    matches = re.findall(r"[A-Za-z0-9]{4,10}", name)
    return max(matches, key=len) if matches else None

def segment_chars(img: np.ndarray, expected_len: int) -> list:
    """Segmenta caracteres da imagem"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontra contornos
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= MIN_CHAR_AREA]
    boxes.sort(key=lambda b: b[0])
    
    # Se não encontrar contornos suficientes, divide igualmente
    if len(boxes) != expected_len:
        h, w = th.shape
        step = w // expected_len
        boxes = [(i*step, 0, step, h) for i in range(expected_len)]
    
    # Processa cada caractere
    chars = []
    for (x, y, w, h) in boxes[:expected_len]:
        char_img = th[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (IMG_WIDTH, IMG_HEIGHT))
        chars.append(char_img)
    return chars

def create_dataset(dirs: list, limit=0) -> tuple:
    """Cria dataset usando tf.data com processamento em lotes"""
    # Encontra todos os arquivos primeiro
    print("Encontrando arquivos de imagem...")
    all_files = []
    for d in dirs:
        all_files.extend(glob.glob(os.path.join(d, "**", "*.*"), recursive=True))
    
    # Limita o número de arquivos se necessário
    if limit > 0 and limit < len(all_files):
        print(f"Limitando processamento a {limit} arquivos dos {len(all_files)} encontrados")
        all_files = all_files[:limit]
    
    print(f"Total de arquivos a processar: {len(all_files)}")
    
    # Contadores para logging
    total_files = len(all_files)
    valid_labels = 0
    valid_images = 0
    total_chars = 0
    
    # Processa em lotes para economizar memória
    images = []
    labels = []
    
    for i, f in enumerate(tqdm(all_files, desc="Processando imagens")):
        # Log de progresso a cada 1000 imagens
        if i > 0 and i % 1000 == 0:
            print(f"Processados {i}/{total_files} arquivos. Caracteres extraídos: {total_chars}")
        
        # Extrai o rótulo do nome do arquivo
        label = find_label(os.path.basename(f))
        if not label:
            continue
        valid_labels += 1
        
        # Carrega a imagem
        try:
            img = cv2.imread(f)
            if img is None:
                continue
            valid_images += 1
            
            # Segmenta os caracteres
            chars = segment_chars(img, len(label))
            if not chars:
                continue
            
            # Processa cada caractere
            for c_img, char in zip(chars, label.upper()):
                if char not in CHARS:
                    continue
                
                # Pré-processamento
                c_img = np.expand_dims(c_img, axis=-1)  # Adiciona canal
                c_img = c_img.astype('float32') / 255.0  # Normaliza
                
                images.append(c_img)
                labels.append(CHARS.index(char))
                total_chars += 1
        except Exception as e:
            # Captura erros durante o processamento da imagem
            print(f"Erro ao processar {f}: {str(e)}")
    
    # Relatório final
    print(f"\nRelatório de Processamento:")
    print(f"Total de arquivos: {total_files}")
    print(f"Arquivos com rótulos válidos: {valid_labels}")
    print(f"Imagens carregadas com sucesso: {valid_images}")
    print(f"Total de caracteres extraídos: {total_chars}")
    
    return images, labels

def build_model() -> tf.keras.Model:
    """Constrói modelo CNN com ~2 milhões de parâmetros, otimizado para velocidade"""
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Data augmentation simples
    x = tf.keras.layers.RandomRotation(0.1)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    
    # Bloco convolucional - usando strides=2 em vez de MaxPooling para acelerar
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Segundo bloco
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Terceiro bloco
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Camadas densas - reduzindo dropout para acelerar
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CHARS), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Otimizador com learning rate mais alto para convergência mais rápida
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def main():
    parser = argparse.ArgumentParser(description='Treina modelo CNN para captcha')
    parser.add_argument('dataset_dirs', nargs='*', help='Diretórios com imagens')
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--limit', type=int, default=0, help='Limite de imagens a processar (0 = sem limite)')
    args = parser.parse_args()
    
    # Registra o início do processamento
    start_time = time.time()
    print(f"Iniciando processamento às {time.strftime('%H:%M:%S')}")
    
    # Prepara dados
    data_dirs = args.dataset_dirs or [str(DEFAULT_DATA_DIR)]
    print(f"Diretórios de dados: {data_dirs}")
    
    # Cria o dataset com limite opcional
    images, labels = create_dataset(data_dirs, args.limit)
    
    if not images:
        print("Nenhuma imagem foi processada! Verifique os diretórios e formatos de arquivo.")
        return
    
    print(f"Convertendo para arrays numpy...")
    # Convert to numpy arrays
    images_np = np.array(images)
    labels_np = np.array(labels)
    
    print(f"Formato dos dados: {images_np.shape}, {labels_np.shape}")
    
    # Verificar se o TensorFlow está usando GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nTreinando com GPU: {gpus}")
        # Configurar para usar toda a memória da GPU conforme necessário
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            print("Não foi possível configurar o crescimento de memória da GPU")
    else:
        print("\nNenhuma GPU encontrada, treinando com CPU")
    
    # Create dataset com otimizações
    print("Criando tf.data.Dataset otimizado...")
    dataset = tf.data.Dataset.from_tensor_slices((images_np, labels_np))
    
    # Otimizações do pipeline de dados
    # Reduzindo o uso de cache para evitar problemas de memória
    dataset = dataset.shuffle(buffer_size=min(len(images_np), 5000))
    dataset = dataset.batch(args.batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch para próximo batch durante processamento
    
    # Split into train and validation - usando menos dados para validação
    train_size = int((1 - VAL_SPLIT) * len(images_np))
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    print(f"Tamanho do conjunto de treinamento: {train_size}")
    print(f"Tamanho do conjunto de validação: {len(images_np) - train_size}")
    
    # Treina modelo
    print("\nConstruindo o modelo CNN...")
    model = build_model()
    model.summary()
    
    # Callbacks simplificados para evitar problemas de compatibilidade
    print("\nConfigurando callbacks simplificados...")
    checkpoint_path = "best_model.h5"
    callbacks = [
        # Salvar apenas o melhor modelo
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_accuracy'
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        # Redução de learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
        # Removido TensorBoard para evitar problemas
    ]
    
    # Treinamento com otimizações
    print(f"\nIniciando treinamento acelerado com {args.epochs} épocas...")
    
    # Removido mixed precision para evitar problemas de compatibilidade
    
    # Treinamento com otimizações básicas - removido parâmetros não suportados
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Carrega os melhores pesos
    if os.path.exists(checkpoint_path):
        print(f"\nCarregando o melhor modelo salvo em {checkpoint_path}...")
        model = tf.keras.models.load_model(checkpoint_path)
    
    # Avalia o modelo
    print("\nAvaliando o modelo no conjunto de validação...")
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Acurácia de validação final: {accuracy:.4f}")
    
    # Save model
    final_model_path = 'captcha_model.h5'
    model.save(final_model_path)
    print(f"\nModelo salvo em {final_model_path}")
    
    # Tempo total
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTempo total de execução: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()