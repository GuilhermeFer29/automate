import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from PIL import Image
import logging
import random
import numpy as np
from tqdm import tqdm
from tqdm.keras import TqdmCallback

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('captcha_solver')

# Configurações do TensorFlow ANTES de qualquer uso
tf.config.optimizer.set_jit(True)
tf.config.set_visible_devices([], 'GPU')  # Desabilita GPU

class AdvancedCaptchaSolver:
    def __init__(self, max_length=10, chars='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'):
        self.max_length = max_length
        self.CHARS = chars
        self.num_chars = len(chars)
        self.char_to_index = {char: idx for idx, char in enumerate(chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(chars)}
        self.model = None
        self.optimizer = optimizers.Adam(learning_rate=0.0001)
        
    def find_image_paths(self, data_dir):
        """Busca recursivamente por imagens com barra de progresso."""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        image_paths = []
        
        logger.info(f"Iniciando varredura de diretórios em: {data_dir}...")
        walker = os.walk(data_dir)
        for root, _, files in tqdm(walker, desc="Varrendo diretórios", unit="dir", ncols=100):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            logger.error(f"Nenhuma imagem encontrada no diretório: {data_dir}. Verifique o caminho e os arquivos.")
        else:
            logger.info(f"{len(image_paths)} imagens encontradas após varredura.")
        return image_paths

    def parse_image(self, data, is_path=True):
        try:
            if is_path:
                path = data
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    return None
                img = Image.open(path)
                label = os.path.splitext(os.path.basename(path))[0]
            else:  # data é uma tupla (base64_string, label)
                import base64
                from io import BytesIO
                b64_data, label = data
                img_data = base64.b64decode(b64_data)
                img = Image.open(BytesIO(img_data))

            # Processamento da imagem
            img = img.convert('L')
            img = np.array(img)
            img = tf.image.resize(img[..., tf.newaxis], [40, 120]).numpy()
            img = img / 255.0

            # Validação e normalização do label (unificado para ambos os formatos)
            if not all(c in self.CHARS for c in label):
                return None  # Ignora silenciosamente para não poluir o log

            if len(label) > self.max_length:
                label = label[:self.max_length]
            else:
                label = label.ljust(self.max_length, '_')

            # Codificar o label
            encoded_label = [self.char_to_index[c] for c in label]
            return img, np.array(encoded_label, dtype=np.int32)
        except Exception:
            return None # Ignora erros de processamento silenciosamente

    def _preprocess_and_cache_data(self, image_paths, cache_path, desc="Preprocessing"):
        """Processa imagens (arquivo/base64) com barra de progresso e salva em cache .npz."""
        logger.info(f"Iniciando pré-processamento e cache para: {desc}...")
        processed_images = []
        processed_labels = []
        import base64

        for path in tqdm(image_paths, desc=desc, unit="img", ncols=100):
            use_base64 = random.random() < 0.5  # 50% de chance
            try:
                if use_base64:
                    with open(path, 'rb') as f:
                        img_bytes = f.read()
                    b64_data = base64.b64encode(img_bytes).decode('utf-8')
                    label_str = os.path.splitext(os.path.basename(path))[0]
                    result = self.parse_image((b64_data, label_str), is_path=False)
                else:
                    result = self.parse_image(path, is_path=True)
                
                if result is not None:
                    img_array, label_array = result
                    processed_images.append(img_array)
                    processed_labels.append(label_array)
            except Exception as e:
                logger.warning(f"Erro ao processar {path}: {e}. Ignorando.")
                continue
        
        if not processed_images: # Se nenhuma imagem foi processada com sucesso
            logger.error(f"Nenhuma imagem processada com sucesso para {desc}. Verifique os caminhos e o formato das imagens.")
            # Retorna arrays vazios com os dtypes corretos para evitar falhas no from_tensor_slices
            return np.array([], dtype=np.float32).reshape(0, 40, 120, 1), np.array([], dtype=np.int32).reshape(0, self.max_length)

        # Converte listas para arrays NumPy
        processed_images_np = np.array(processed_images, dtype=np.float32)
        processed_labels_np = np.array(processed_labels, dtype=np.int32)

        # Garante que o diretório de cache exista
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, images=processed_images_np, labels=processed_labels_np)
        logger.info(f"Dados de '{desc}' processados e salvos em cache: {cache_path}")
        return processed_images_np, processed_labels_np

    def create_dataset(self, image_paths, batch_size, shuffle, cache_filename, desc):
        """Cria dataset a partir de dados cacheados ou processa e cacheia se necessário."""
        # Define um diretório de cache dentro do diretório do script ou um local mais apropriado
        cache_dir = os.path.join(os.path.dirname(__file__), '.captcha_cache') 
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path):
            logger.info(f"Carregando dados pré-processados de '{desc}' do cache: {cache_path}")
            try:
                data = np.load(cache_path)
                images_np = data['images']
                labels_np = data['labels']
                if images_np.size == 0 or labels_np.size == 0:
                    logger.warning(f"Cache {cache_path} está vazio ou corrompido. Re-processando...")
                    images_np, labels_np = self._preprocess_and_cache_data(image_paths, cache_path, desc=desc)
            except Exception as e:
                logger.warning(f"Erro ao carregar cache {cache_path}: {e}. Re-processando...")
                images_np, labels_np = self._preprocess_and_cache_data(image_paths, cache_path, desc=desc)
        else:
            logger.info(f"Cache não encontrado para '{desc}'. Iniciando pré-processamento...")
            images_np, labels_np = self._preprocess_and_cache_data(image_paths, cache_path, desc=desc)

        if images_np.size == 0: # Verifica se, após tudo, ainda não há dados
             logger.error(f"Falha crítica: Não há dados para criar o dataset '{desc}'. Verifique a fonte de dados.")
             # Cria um dataset vazio para evitar falhas no TensorFlow, mas o treinamento não ocorrerá.
             # Isso pode precisar de um tratamento mais robusto dependendo do fluxo desejado.
             empty_images = tf.zeros([0, 40, 120, 1], dtype=tf.float32)
             empty_labels_dict = {f'char_{i}': tf.zeros([0], dtype=tf.int32) for i in range(self.max_length)}
             return tf.data.Dataset.from_tensor_slices((empty_images, empty_labels_dict)).batch(batch_size)

        dataset = tf.data.Dataset.from_tensor_slices((images_np, labels_np))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images_np), reshuffle_each_iteration=True)

        def format_output(img, label):
            return img, {f'char_{i}': label[i] for i in range(self.max_length)}

        dataset = dataset.map(format_output, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.info(f"Dataset '{desc}' criado e pronto para uso.")
        return dataset

    def build_optimized_model(self):
        """Constrói modelo com regularização e batch normalization"""
        logger.info("ETAPA 3/5: Construindo o modelo...")
        input_layer = layers.Input(shape=(40, 120, 1), name='input')
        
        # Bloco Convolucional 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Bloco Convolucional 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Bloco Convolucional 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Camadas Densas
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Múltiplas saídas (uma para cada caractere)
        outputs = [
            layers.Dense(len(self.CHARS), activation='linear', name=f'char_{i}')(x)
            for i in range(self.max_length)
        ]
        
        self.model = models.Model(inputs=input_layer, outputs=outputs)
        
        # Compilar modelo
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            optimizer=self.optimizer,
            loss={f'char_{i}': loss_function for i in range(self.max_length)},
            metrics={f'char_{i}': ['accuracy'] for i in range(self.max_length)}
        )
        
        logger.info("ETAPA 3/5: Construção do modelo concluída.")
        self.model.summary()

    def train_model(self, data_dir, epochs=30, batch_size=128, validation_split=0.2, model_save_path='saved_model'):
        """Treina o modelo com callbacks avançados"""
        logger.info("ETAPA 1/5: Buscando imagens...")
        image_paths = self.find_image_paths(data_dir)
        if not image_paths:
            raise ValueError("Nenhuma imagem encontrada no diretório fornecido")
        logger.info("ETAPA 1/5: Busca de imagens concluída.")
        
        # Dividir em treino e validação
        split_idx = int(len(image_paths) * (1 - validation_split))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
        logger.info(f"Dataset dividido em {len(train_paths)} para treino e {len(val_paths)} para validação.")
        
        logger.info("ETAPA 2/5: Criando datasets de treino e validação...")
        logger.info("ETAPA 2/5: Preparando datasets de treino e validação (usando cache se disponível)...")
        train_ds = self.create_dataset(train_paths, batch_size, shuffle=True, cache_filename="train_data_cache.npz", desc="Treino")
        val_ds = self.create_dataset(val_paths, batch_size, shuffle=False, cache_filename="validation_data_cache.npz", desc="Validação")
        logger.info("ETAPA 2/5: Criação de datasets concluída.")
        
        # O modelo deve ser construído fora desta função
        if self.model is None:
            logger.error("O modelo não foi construído. Chame `build_optimized_model()` antes de treinar.")
            return None

        logger.info("ETAPA 4/5: Configurando callbacks...")
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(model_save_path, 'model_epoch_{epoch:02d}.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(model_save_path, 'logs'),
                histogram_freq=1
            ),
            callbacks.BackupAndRestore(
                backup_dir=os.path.join(model_save_path, 'backup')
            ),
            TqdmCallback(verbose=2)  # Barra de progresso detalhada e em tempo real para cada época
        ]
        logger.info("ETAPA 4/5: Configuração de callbacks concluída.")
        
        logger.info("ETAPA 5/5: Iniciando o treinamento do modelo...")
        # Treinar modelo
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=0  # Desativa o log padrão para usar o TQDM
        )
        logger.info("ETAPA 5/5: Treinamento concluído.")
        
        # Salvar modelo final
        self.model.save(os.path.join(model_save_path, 'final_model.h5'))
        logger.info(f"Modelo salvo em {model_save_path}")
        
        return history

    def predict_captcha(self, image_path):
        """Prediz CAPTCHA a partir de um caminho de imagem"""
        if not self.model:
            raise ValueError("Modelo não foi treinado")
            
        img, _ = self.parse_image(image_path)
        if img is None:
            raise ValueError("Imagem inválida ou não pôde ser processada")
            
        # Adicionar dimensão de batch
        img = np.expand_dims(img, axis=0)
        
        # Fazer predição
        predictions = self.model.predict(img)
        
        # Decodificar predição
        captcha_text = ''
        for i in range(self.max_length):
            pred_idx = np.argmax(predictions[i][0])
            captcha_text += self.CHARS[pred_idx]
            
        return captcha_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Treina modelo para resolver CAPTCHAs')
    parser.add_argument('--data_dir', required=True, help='Diretório com imagens CAPTCHA')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch')
    parser.add_argument('--model_save_path', default='captcha_model', help='Caminho para salvar o modelo')
    
    args = parser.parse_args()
    
    solver = AdvancedCaptchaSolver(max_length=10)
    solver.build_optimized_model()
    
    logger.info(f"Iniciando treinamento com {args.epochs} épocas...")
    solver.train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )