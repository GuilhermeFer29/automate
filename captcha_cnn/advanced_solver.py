import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow
import cv2
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
# Força o pipeline tf.data a rodar em modo eager.
try:
    tf.data.experimental.enable_debug_mode()
except AttributeError:
    # Versões antigas do TensorFlow não possuem debug_mode
    pass

from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import string
from collections import Counter
import time
import glob
import os
import tensorflow as tf
from tqdm import tqdm  # Correção da importação
from tqdm.keras import TqdmCallback
import uuid

# Configuração de recursos
os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
tf.config.set_soft_device_placement(True)

physical_devices = tf.config.list_physical_devices('CPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEPOCHA {epoch + 1} COMPLETA")
        print("="*40)
        for k, v in logs.items():
            print(f"{k.upper()}: {v:.4f}")
        print("="*40 + "\n")

class AdvancedCaptchaSolver:
    def __init__(self, img_width=180, img_height=50, max_length=5):
        # Configuração agressiva de memória
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        self.checkpoint_dir = os.path.join('checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length  # será atualizado dinamicamente na criação do dataset
        
        # Caracteres dinâmicos com especiais
        self.CHARS = self.get_all_chars('/home/guilherme/Documentos/GitHub/automate/captcha_cnn/imagens')
        self.num_chars = len(self.CHARS)
        self.model = None
        self.total_params = 0
        self.optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.callback_list = [
            callbacks.EarlyStopping(
                monitor='val_char_0_accuracy',
                patience=15,
                mode='max',
                restore_best_weights=True),
                
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_char_0_accuracy',
                save_best_only=True,
                mode='max'),
                
            callbacks.TensorBoard(
                log_dir='./logs',
                profile_batch=0,
                update_freq='batch'),
                
            callbacks.BackupAndRestore('./backup')
        ]

    def get_all_chars(self, data_dir):
        chars = set()
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.split('.')[-1].lower() in ['jpg','png','jpeg']:
                    chars.update(list(os.path.splitext(file)[0]))
        return sorted(chars)

    def preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
                
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.equalizeHist(img)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
            return np.expand_dims(img.astype(np.float32)/255.0, axis=-1)
        except Exception as e:
            print(f"Erro processando {img_path}: {str(e)}")
            return None
    
    def load_dataset(self, data_dir):
        """Carrega dataset com verificação rigorosa de dimensões"""
        images, labels = [], []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = self.preprocess_image(os.path.join(root, file))
                        label = os.path.splitext(file)[0]
                        
                        if img is not None and len(label) == self.max_length:
                            # Garante label no formato correto
                            encoded_label = [self.CHARS.index(c) for c in label if c in self.CHARS]
                            if len(encoded_label) == self.max_length:
                                images.append(img)
                                labels.append(encoded_label)
                    except Exception as e:
                        print(f"Arquivo {file} ignorado: {str(e)}")
        
        # Verificação final de dimensões
        if not images:
            raise ValueError("Nenhuma imagem válida encontrada")
            
        return np.array(images), np.array(labels)
    
    def build_optimized_model(self):
        """Modelo corrigido com conexões adequadas"""
        inputs = layers.Input(shape=(40, 120, 1))  # Dimensões atualizadas
        x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(64, (3,3), activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        output_layers = {}
        for i in range(self.max_length):
            output_layers[f'char_{i}'] = layers.Dense(self.num_chars, activation='softmax', name=f'char_{i}')(x)
        
        self.model = models.Model(inputs=inputs, outputs=output_layers)
        self.model.compile(
            optimizer=self.optimizer,
            loss={name: 'sparse_categorical_crossentropy' for name in output_layers.keys()},
            metrics={name: 'accuracy' for name in output_layers.keys()}
        )
        
        self.total_params = self.model.count_params() / 1e6
        print(f"Modelo construído com {self.total_params:.2f}M parâmetros")
        return self.model

    def build_model(self):
        input_img = tf.keras.Input(shape=(50, 180, 1), name='image_input')
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(input_img)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # 6 saídas independentes (uma para cada caractere)
        outputs = [tf.keras.layers.Dense(len(self.CHARS), activation='softmax', name=f'char_{i}')(x) 
                   for i in range(6)]
        
        self.model = tf.keras.Model(inputs=input_img, outputs=outputs)
        
        # Loss e métricas para cada caractere
        losses = {f'char_{i}': 'sparse_categorical_crossentropy' for i in range(6)}
        metrics = {f'char_{i}': 'accuracy' for i in range(6)}
        
        self.model.compile(
            optimizer='adam',
            loss=losses,
            metrics=metrics
        )

    def find_image_files(self, data_dir):
        """Busca recursiva por arquivos PNG válidos"""
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, file))
        return image_files

    def _parse_label(self, file_name):
        """Processa labels no formato 'XXXX_YYYY' extraindo apenas os caracteres válidos"""
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        
        # Extrai apenas os caracteres antes do '_' (se existir)
        label_part = base_name.split('_')[0] if '_' in base_name else base_name
        
        # Filtra caracteres válidos
        valid_chars = [c for c in label_part if c in self.CHARS]
        if not valid_chars:
            raise ValueError(f"Nenhum caractere válido encontrado em {base_name}")
        
        return [self.CHARS.index(c) for c in valid_chars]

    def _process_sample(self, path):
        """Processa um sample individual com tratamento de erros"""
        try:
            label = self._parse_label(path)
            return (path, label)
        except ValueError as e:
            print(f"Arquivo ignorado {os.path.basename(path)}: {str(e)}")
            return None

    def create_dataset(self, data_dir, batch_size=128):
        print("\n🔍 Fase 1/4: Varredura de arquivos...")
        samples = []
        for root, _, files in tqdm(os.walk(data_dir), desc="Progresso"):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    # ignora arquivos vazios / corrompidos
                    try:
                        if os.path.getsize(full_path) == 0:
                            continue
                    except OSError:
                        continue
                    label = os.path.basename(file).split('_')[0]
                    samples.append((full_path, label))
                    if len(samples) % 5000 == 0:
                        print(f"✓ {len(samples)} arquivos processados")

        print("\n🔍 Fase 2/4: Análise de caracteres...")
        all_chars = set()
        for _, label in tqdm(samples, desc="Processando"):
            all_chars.update(label)
        
        # Adiciona caractere de padding '-' se ainda não existir
        all_chars.add('-')
        self.CHARS = sorted(all_chars)
        self.num_chars = len(self.CHARS)
        print(f"\n🔤 Caracteres únicos ({len(self.CHARS)}): {''.join(self.CHARS)}")
        
        char_to_idx = {char: idx for idx, char in enumerate(self.CHARS)}
        max_len = max(len(label) for _, label in samples)
        # Atualiza dinamicamente self.max_length para maior valor encontrado
        if max_len != self.max_length:
            print(f"ℹ️  max_length atualizado de {self.max_length} para {max_len}")
            self.max_length = max_len
        target_len = self.max_length
        
        print(f"\n🔍 Fase 3/4: Processando labels (max_len={max_len})...")
        labels = []
        for path, label in tqdm(samples, desc="Convertendo"):
            padded = label.ljust(target_len, '-')  # usa '-' como padding temporário
            encoded = [char_to_idx[c] for c in padded]
            labels.append(encoded)
        
        print("\n🔍 Fase 4/4: Criando dataset final (streaming) ...")
        # Constroi tensores para streaming
        labels_dict = {f'char_{i}': tf.constant([lab[i] for lab in labels], dtype=tf.int32) for i in range(self.max_length)}
        paths_tensor = tf.constant([s[0] for s in samples])

        def _parse_fn(path, *label_values):
            raw = tf.io.read_file(path)
            img = tf.image.decode_png(raw, channels=1)
            img = tf.image.resize(img, [40, 120], method='nearest')  # Reduzir dimensões
            img = tf.image.convert_image_dtype(img, tf.float16)
            label_dict = {f'char_{i}': label_values[i] for i in range(self.max_length)}
            return img, label_dict

        # Dataset de caminhos + labels transposto para lista
        label_tensors = [labels_dict[f'char_{i}'] for i in range(self.max_length)]
        cache_dir = f'/tmp/captcha_cache_{uuid.uuid4()}'
        dataset = (
            tf.data.Dataset.from_tensor_slices((paths_tensor, *label_tensors))
              .shuffle(buffer_size=10000)  # Buffer menor para economizar RAM
              .map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
              .cache(cache_dir)
              .batch(batch_size)
              .prefetch(2)  # Prefetch menor para evitar sobrecarga
        )
        return dataset

    def train_model(self, data_dir, epochs=30, batch_size=128):

        
        # Contagem direta de arquivos PNG
        pattern = os.path.join(data_dir, '**', '*.png')
        file_list = glob.glob(pattern, recursive=True)
        num_samples = len(file_list)
        steps_per_epoch = max(1, num_samples // batch_size)
        print(f"\n📊 Steps por época: {steps_per_epoch} (Total samples: {num_samples})")

        
        # Criar dataset APÓS a contagem
        train_ds = self.create_dataset(data_dir, batch_size)

        # Garante que o modelo possui o mesmo número de saídas que max_length
        if self.model is None or len(self.model.outputs) != self.max_length:
            print(f"🔄 Reconstruindo modelo para {self.max_length} saídas...")
            self.build_optimized_model()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'cp-{epoch:04d}.weights.h5'),
                save_weights_only=True,
                verbose=0),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_char_0_accuracy',
                patience=15,
                mode='max',
                restore_best_weights=True),
            TqdmCallback(verbose=1)
        ]
        
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=0
        )
        

        return history

    def evaluate_model(self, test_dir):
        """Avaliação detalhada do modelo"""
        X_test, y_test = self.load_dataset(test_dir)
        
        print("\n=== Avaliação do Modelo ===")
        results = self.model.evaluate(
            X_test, 
            {f'char_{i}': y_test[:,i] for i in range(self.max_length)},
            verbose=1
        )
        
        # Calcula acurácia por posição de caractere
        predictions = self.model.predict(X_test)
        for i in range(self.max_length):
            correct = np.sum(np.argmax(predictions[f'char_{i}'], axis=1) == y_test[:,i])
            acc = correct / len(y_test)
            print(f"Acurácia posição {i+1}: {acc:.2%}")
        
        return results

    def evaluate_on_real_captchas(self, test_dir, threshold=0.8):
        """Avaliação em CAPTCHAs reais com limiar de confiança"""
        import glob
        
        print(f"\n=== Avaliação em CAPTCHAs Reais ===")
        captcha_files = glob.glob(os.path.join(test_dir, '*.png')) + \
                       glob.glob(os.path.join(test_dir, '*.jpg'))
        
        total = len(captcha_files)
        correct = 0
        
        for file in captcha_files:
            img = self.preprocess_image(file)
            if img is None:
                continue
                
            preds = self.model.predict(np.expand_dims(img, axis=0))
            predicted = ''
            confidences = []
            
            for i in range(self.max_length):
                char_probs = preds[f'char_{i}'][0]
                max_idx = np.argmax(char_probs)
                confidence = char_probs[max_idx]
                if confidence > threshold:
                    predicted += self.CHARS[max_idx]
                confidences.append(confidence)
            
            true_label = os.path.splitext(os.path.basename(file))[0]
            true_label = ''.join(c for c in true_label if c in self.CHARS)
            
            if predicted == true_label:
                correct += 1
            
            print(f"Arquivo: {os.path.basename(file)}")
            print(f"Previsto: {predicted} (Confiança média: {np.mean(confidences):.2f})")
            print(f"Verdadeiro: {true_label}")
            print("-" * 50)
        
        accuracy = correct / total
        print(f"\nAcurácia em CAPTCHAs reais: {accuracy:.2%} ({correct}/{total})")
        return accuracy


def test_dataset_pipeline():
    solver = AdvancedCaptchaSolver()
    test_ds = solver.create_dataset('imagens', batch_size=2)
    
    print("\n🔬 Testando pipeline:")
    for images, labels in test_ds.take(1):
        print(f"Batch shapes - Images: {images.shape}, Labels: {labels}")
        print(f"Primeira imagem - Min: {tf.reduce_min(images[0])}, Max: {tf.reduce_max(images[0])}")
        print(f"Primeiros labels: { {k: v.numpy()[0] for k, v in labels.items()} }")

def test_model_compatibility():
    solver = AdvancedCaptchaSolver()
    solver.build_model()
    
    # Teste com tensor dummy
    dummy_img = tf.random.uniform((1, 50, 180, 1))
    output = solver.model.predict(dummy_img)
    
    print("\n🧠 Teste de compatibilidade do modelo:")
    print(f"Output shapes: {[o.shape for o in output]}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Treina o modelo CAPTCHA CNN")
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do lote')
    args = parser.parse_args()
    # Configurações com verificação
    base_dir = '/home/guilherme/Documentos/GitHub/automate/captcha_cnn/imagens'
    
    # Lista todos os subdiretórios que contêm imagens
    image_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            image_dirs.append(root)
    
    if not image_dirs:
        raise FileNotFoundError(f"Nenhum diretório com imagens encontrado em {base_dir}")
    
    print("Diretórios com imagens encontrados:")
    for i, dir_path in enumerate(image_dirs[:5]):
        print(f"{i+1}. {dir_path}")
    
    # Usa o primeiro diretório com imagens encontrado
    data_dir = image_dirs[0]
    print(f"\nUsando diretório: {data_dir}")
    
    # Inicializa solver
    solver = AdvancedCaptchaSolver()
    
    # Constrói modelo explicitamente
    print("Construindo modelo...")
    solver.build_model()
    
    # Resumo do modelo
    solver.model.summary()
    
    # Treinamento com novos parâmetros
    print("\nIniciando treinamento...")
    history = solver.train_model(
        data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )  
    # Avaliação
    print("\nAvaliando modelo...")
    accuracy = solver.evaluate_model(data_dir)
    print(f"\nAcurácia final: {accuracy:.2%}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test_dataset_pipeline()
        test_model_compatibility()
        sys.exit(0)
    main()
