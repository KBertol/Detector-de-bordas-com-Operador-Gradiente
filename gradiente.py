import cv2
import numpy as np
from skimage import feature
from skimage.metrics import structural_similarity as ssim

def aplicar_convolucao_manual(imagem, kernel):

    altura_imagem, largura_imagem = imagem.shape
    altura_kernel, largura_kernel = kernel.shape

    borda_y = altura_kernel // 2
    borda_x = largura_kernel // 2

    imagem_com_borda = np.zeros((altura_imagem + (borda_y * 2), largura_imagem + (borda_x * 2)))
    imagem_com_borda[borda_y:-borda_y, borda_x:-borda_x] = imagem

    imagem_convolucionada = np.zeros((altura_imagem, largura_imagem))

    for i in range(altura_imagem):
        for j in range(largura_imagem):
            regiao_interesse = imagem_com_borda[i : i + altura_kernel, j : j + largura_kernel]
            soma_convolucao = np.sum(regiao_interesse * kernel)
            imagem_convolucionada[i, j] = soma_convolucao

    return imagem_convolucionada

def obter_gradientes(imagem, operador="prewitt"):
    if operador == "prewitt":
        kernel_x = np.array([[-1,  0,  1],
                             [-1,  0,  1],
                             [-1,  0,  1]], dtype=np.float64)

        kernel_y = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]], dtype=np.float64)

    elif operador == "scharr":
        kernel_x = np.array([[ -3,  0,   3],
                             [-10,  0,  10],
                             [ -3,  0,   3]], dtype=np.float64)

        kernel_y = np.array([[ -3, -10,  -3],
                             [  0,   0,   0],
                             [  3,  10,   3]], dtype=np.float64)
    else:
        raise ValueError("Operador inválido. Use 'prewitt' ou 'scharr'.")

    gradiente_x = aplicar_convolucao_manual(imagem, kernel_x)
    gradiente_y = aplicar_convolucao_manual(imagem, kernel_y)

    return gradiente_x, gradiente_y

def calcular_magnitude_e_direcao(gradiente_x, gradiente_y):
    matriz_magnitude = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)

    matriz_direcao_graus = np.degrees(np.arctan2(gradiente_y, gradiente_x))

    matriz_direcao_graus[matriz_direcao_graus < 0] += 180

    return matriz_magnitude, matriz_direcao_graus

def suprimir_nao_maximos_b1(matriz_magnitude, matriz_direcao):
    altura, largura = matriz_magnitude.shape
    imagem_suprimida = np.zeros((altura, largura))

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            angulo = matriz_direcao[i, j]
            magnitude_atual = matriz_magnitude[i, j]

            if (0 <= angulo <= 22.5) or (157.5 < angulo <= 180):
                # Direção horizontal: vizinhos D e F
                vizinho_1 = matriz_magnitude[i, j + 1]  # F
                vizinho_2 = matriz_magnitude[i, j - 1]  # D

            elif (22.5 < angulo <= 67.5):
                # Diagonal 45°: vizinhos C e G
                vizinho_1 = matriz_magnitude[i - 1, j + 1]  # C
                vizinho_2 = matriz_magnitude[i + 1, j - 1]  # G

            elif (67.5 < angulo <= 112.5):
                # Direção vertical: vizinhos B e H
                vizinho_1 = matriz_magnitude[i - 1, j]  # B
                vizinho_2 = matriz_magnitude[i + 1, j]  # H

            elif (112.5 < angulo <= 157.5):
                # Diagonal 135°: vizinhos A e I
                vizinho_1 = matriz_magnitude[i - 1, j - 1]  # A
                vizinho_2 = matriz_magnitude[i + 1, j + 1]  # I

            else:
                vizinho_1 = 0
                vizinho_2 = 0

            if (magnitude_atual > vizinho_1) and (magnitude_atual > vizinho_2):
                imagem_suprimida[i, j] = magnitude_atual
            else:
                imagem_suprimida[i, j] = 0

    return imagem_suprimida

def suprimir_nao_maximos_b2(matriz_magnitude, gradiente_x, gradiente_y):
    altura, largura = matriz_magnitude.shape
    imagem_suprimida = np.zeros((altura, largura))

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            gx = gradiente_x[i, j]
            gy = gradiente_y[i, j]
            magnitude_atual = matriz_magnitude[i, j]

            abs_gx = abs(gx)
            abs_gy = abs(gy)

            A = matriz_magnitude[i - 1, j - 1]
            B = matriz_magnitude[i - 1, j    ]
            C = matriz_magnitude[i - 1, j + 1]
            D = matriz_magnitude[i,     j - 1]
            F = matriz_magnitude[i,     j + 1]
            G = matriz_magnitude[i + 1, j - 1]
            H = matriz_magnitude[i + 1, j    ]
            I = matriz_magnitude[i + 1, j + 1]

            if gx >= 0 and gy >= 0:
                if abs_gx >= abs_gy:
                    t = abs_gy / (abs_gx + 1e-8)
                    vizinho_Y = (1 - t) * F + t * C   # entre F e C
                    vizinho_X = (1 - t) * D + t * G   # entre D e G
                else:
                    t = abs_gx / (abs_gy + 1e-8)
                    vizinho_Y = (1 - t) * B + t * C   # entre B e C
                    vizinho_X = (1 - t) * H + t * G   # entre H e G

            elif gx < 0 and gy >= 0:
                if abs_gy >= abs_gx:
                    t = abs_gx / (abs_gy + 1e-8)
                    vizinho_Y = (1 - t) * B + t * A   # entre B e A
                    vizinho_X = (1 - t) * H + t * I   # entre H e I
                else:
                    t = abs_gy / (abs_gx + 1e-8)
                    vizinho_Y = (1 - t) * D + t * A   # entre D e A
                    vizinho_X = (1 - t) * F + t * I   # entre F e I

            elif gx < 0 and gy < 0:
                if abs_gx >= abs_gy:
                    t = abs_gy / (abs_gx + 1e-8)
                    vizinho_Y = (1 - t) * D + t * G   # entre D e G
                    vizinho_X = (1 - t) * F + t * C   # entre F e C
                else:
                    t = abs_gx / (abs_gy + 1e-8)
                    vizinho_Y = (1 - t) * H + t * G   # entre H e G
                    vizinho_X = (1 - t) * B + t * C   # entre B e C

            else:  
                if abs_gx >= abs_gy:
                    t = abs_gy / (abs_gx + 1e-8)
                    vizinho_Y = (1 - t) * F + t * I   # entre F e I
                    vizinho_X = (1 - t) * D + t * A   # entre D e A
                else:
                    t = abs_gx / (abs_gy + 1e-8)
                    vizinho_Y = (1 - t) * H + t * I   # entre H e I
                    vizinho_X = (1 - t) * B + t * A   # entre B e A

            if (magnitude_atual > vizinho_Y) and (magnitude_atual > vizinho_X):
                imagem_suprimida[i, j] = magnitude_atual
            else:
                imagem_suprimida[i, j] = 0

    return imagem_suprimida

def calcular_limiar_adaptativo(bordas_afinadas, fator=1.5):
    valores_nao_nulos = bordas_afinadas[bordas_afinadas > 0]

    if valores_nao_nulos.size == 0:
        return 0

    mediana = np.median(valores_nao_nulos)
    limiar = fator * mediana

    # Evita que o limiar "exploda" além do maior valor observado,
    # o que pode acontecer em imagens com distribuição de magnitude
    # muito concentrada/enviesada (ex.: imagens binárias de alto
    # contraste, como tabuleiros de xadrez), zerando todas as bordas.
    maximo = valores_nao_nulos.max()
    limiar = min(limiar, maximo * 0.8)

    return limiar

def comparar_com_canny_e_ssim(imagem_suavizada, sua_imagem_binarizada, nome_imagem, operador, sigma=1.0):
    bordas_canny_bool = feature.canny(imagem_suavizada, sigma=sigma)
    bordas_canny = np.where(bordas_canny_bool, 255, 0).astype(np.uint8)

    score_ssim, _ = ssim(sua_imagem_binarizada, bordas_canny, full=True)
    print(f"[{nome_imagem} | {operador.upper()} | Canny sigma={sigma}] Índice SSIM: {score_ssim:.4f}")

    return bordas_canny

def processar_lote_de_imagens(lista_imagens, operador_escolhido="prewitt", sigmas_canny=(1.0, 3.0)):
    for arquivo in lista_imagens:
        print(f"\n--- Iniciando: {arquivo} com {operador_escolhido.upper()} ---")

        imagem_original = cv2.imread(arquivo)
        if imagem_original is None:
            print(f"Aviso: O arquivo '{arquivo}' não foi encontrado no diretório.")
            continue

        imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

        imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (3, 3), 0)

        imagem_float = imagem_suavizada.astype(np.float64)

        gradiente_x, gradiente_y = obter_gradientes(imagem_float, operador_escolhido)
        magnitude, direcao = calcular_magnitude_e_direcao(gradiente_x, gradiente_y)

        bordas_b1 = suprimir_nao_maximos_b1(magnitude, direcao)
        limiar_b1 = calcular_limiar_adaptativo(bordas_b1, fator=1.5)
        print(f"Limiar adaptativo (b1): {limiar_b1:.2f}")
        imagem_binarizada_b1 = np.where(bordas_b1 > limiar_b1, 255, 0).astype(np.uint8)

        bordas_b2 = suprimir_nao_maximos_b2(magnitude, gradiente_x, gradiente_y)
        limiar_b2 = calcular_limiar_adaptativo(bordas_b2, fator=1.5)
        print(f"Limiar adaptativo (b2): {limiar_b2:.2f}")
        imagem_binarizada_b2 = np.where(bordas_b2 > limiar_b2, 255, 0).astype(np.uint8)

        nome_saida_b1 = f"resultado_{operador_escolhido}_b1_{arquivo}"
        cv2.imwrite(nome_saida_b1, imagem_binarizada_b1)
        print(f"Salvo: {nome_saida_b1}")

        nome_saida_b2 = f"resultado_{operador_escolhido}_b2_{arquivo}"
        cv2.imwrite(nome_saida_b2, imagem_binarizada_b2)
        print(f"Salvo: {nome_saida_b2}")

        # Comparação com Canny para diferentes valores de sigma
        for sigma in sigmas_canny:
            bordas_canny = comparar_com_canny_e_ssim(
                imagem_suavizada, imagem_binarizada_b1, arquivo, operador_escolhido, sigma=sigma
            )
            comparar_com_canny_e_ssim(
                imagem_suavizada, imagem_binarizada_b2, arquivo, operador_escolhido, sigma=sigma
            )

            sigma_str = str(sigma).replace(".", "_")
            cv2.imwrite(f"resultado_canny_sigma{sigma_str}_{arquivo}", bordas_canny)

if __name__ == "__main__":
    imagens_teste = ["moedas.png", "Lua1_gray.jpg", "chessboard_inv.png", "img2.jpg"]

    processar_lote_de_imagens(imagens_teste, operador_escolhido="prewitt")
    processar_lote_de_imagens(imagens_teste, operador_escolhido="scharr")
