import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()  # Melhora performance em CPU
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

# Configuração de recursos
os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)
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
        self.max_length = max_length
        
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
        inputs = layers.Input(shape=(self.img_height, self.img_width, 1))
        
        # Blocos convolucionais (mantido igual)
        x = layers.Conv2D(48, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(0.25)(x)
        
        x = layers.Conv2D(96, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(0.3)(x)
        
        x = layers.Conv2D(192, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(192, (3,3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(0.4)(x)
        
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dense(384, activation='selu', kernel_initializer='lecun_normal')(x)
        x = layers.Dropout(0.5)(x)
        
        # Saídas conectadas corretamente
        output_layers = {}
        for i in range(self.max_length):
            output_layers[f'char_{i}'] = layers.Dense(self.num_chars, activation='softmax', name=f'char_{i}')(x)
        
        self.model = models.Model(inputs=inputs, outputs=output_layers)
        
        # Compilação
        self.model.compile(
            optimizer=self.optimizer,
            loss={name: 'sparse_categorical_crossentropy' for name in output_layers.keys()},
            metrics=['accuracy'],
            weighted_metrics=['accuracy']
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
                    label = os.path.basename(file).split('_')[0]
                    samples.append((os.path.join(root, file), label))
                    if len(samples) % 5000 == 0:
                        print(f"✓ {len(samples)} arquivos processados")

        print("\n🔍 Fase 2/4: Análise de caracteres...")
        all_chars = set()
        for _, label in tqdm(samples, desc="Processando"):
            all_chars.update(label)
        
        self.CHARS = sorted(all_chars)
        print(f"\n🔤 Caracteres únicos ({len(self.CHARS)}): {''.join(self.CHARS)}")
        
        char_to_idx = {char: idx for idx, char in enumerate(self.CHARS)}
        max_len = max(len(label) for _, label in samples)
        
        print(f"\n🔍 Fase 3/4: Processando labels (max_len={max_len})...")
        labels = []
        for path, label in tqdm(samples, desc="Convertendo"):
            encoded = [char_to_idx.get(c, -1) for c in label] + [-1]*(max_len - len(label))
            labels.append(encoded)
        
        print("\n🔍 Fase 4/4: Criando dataset final...")
        labels_dict = {
            f'char_{i}': tf.constant([label[i] for label in labels])
            for i in range(5)
        }
        
        def load_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=1, expand_animations=False)
            return tf.image.convert_image_dtype(image, tf.float32)
        
        paths = [s[0] for s in samples]
        images = tf.stack([load_image(p) for p in tqdm(paths, desc="Carregando")])
        images = tf.image.resize(images, [50, 180])
        
        dataset = tf.data.Dataset.from_tensor_slices((images, labels_dict))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def train_model(self, data_dir, epochs=30, batch_size=128):
        tf.config.run_functions_eagerly(True)
        
        train_ds = self.create_dataset(data_dir, batch_size)
        
        # Cálculo definitivo e seguro de steps_per_epoch
        num_samples = sum(1 for _ in train_ds.unbatch())
        steps_per_epoch = max(1, num_samples // batch_size)
        print(f"\n📊 Steps por época: {steps_per_epoch} (Total samples: {num_samples})")
        
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
            CustomCallback()
        ]
        
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=0
        )
        
        tf.config.run_functions_eagerly(False)
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

def main():
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
        epochs=30,       # Aumentado para 30 épocas
        batch_size=128   # Aumentado para 128
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
