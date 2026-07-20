# Operador Gradiente na Detecção de Bordas

Implementação própria (sem funções prontas de convolução) dos
operadores de **Prewitt** e **Scharr** para detecção de bordas em
imagens em tons de cinza, com dois métodos de supressão de
não-máximos e comparação quantitativa contra o detector de **Canny**
via índice **SSIM**.

Trabalho prático de **Processamento Digital de Imagens**, desenvolvido
em dupla.

## Pipeline

```
Imagem (RGB/cinza)
        │
        ▼
Filtro Gaussiano 3×3 (redução de ruído)
        │
        ▼
Convolução manual (Gx, Gy) — Prewitt ou Scharr
        │
        ▼
Magnitude M(i,j) = √(Gx² + Gy²)  e  Direção θ(i,j) = atan2(Gy, Gx)
        │
        ▼
Supressão de não-máximos (b.1 ou b.2)
        │
        ▼
Binarização (limiar adaptativo)
        │
        ▼
Imagem de bordas final
        │
        ▼
Comparação com Canny (SSIM, sigma = 1.0 e 3.0)
```

## Detalhes da implementação

- **Convolução manual**: `aplicar_convolucao_manual` implementa a
  convolução 2D (com flip do kernel e padding) do zero, sem usar
  `cv2.filter2D` ou equivalente.
- **Operadores de borda**: kernels de **Prewitt** e **Scharr** para
  Gx e Gy.
- **Supressão de não-máximos**, em duas variantes:
  - **b.1** — compara a magnitude de cada pixel com os dois vizinhos
    colineares à direção do gradiente, escolhidos entre os 8 vizinhos
    mais próximos conforme o octante de θ.
  - **b.2** — em vez de vizinhos discretos, interpola linearmente
    entre dois pixels adjacentes usando como peso a razão entre as
    componentes Gx e Gy, dando uma estimativa mais precisa do valor
    do gradiente exatamente na direção θ.
- **Limiar adaptativo**: `limiar = fator * mediana` das magnitudes
  não-nulas após a supressão, com `fator = 1.5`. Para imagens de alto
  contraste com distribuição de magnitude muito concentrada (ex.:
  `chessboard_inv.png`), esse cálculo pode gerar um limiar maior que a
  própria magnitude máxima e zerar todas as bordas — detectado
  automaticamente pelo coeficiente de variação (`CV = desvio / mediana
  < 0.3`), caso em que o fator é reduzido para `0.5` sem precisar de
  ajuste manual por imagem.
- **Avaliação**: cada resultado é comparado com `skimage.feature.canny`
  (sigma = 1.0 e 3.0) usando o índice **SSIM**
  (`skimage.metrics.structural_similarity`).

## Resultados (SSIM vs. Canny, sigma = 1.0)

| Imagem | Prewitt b1 | Prewitt b2 | Scharr b1 | Scharr b2 |
|---|---|---|---|---|
| moedas.png | 0.5945 | 0.6116 | 0.5883 | 0.6068 |
| Lua1_gray.jpg | 0.9054 | 0.8152 | 0.7990 | 0.7152 |
| chessboard_inv.png | 0.8789 | 0.8797 | 0.8802 | 0.8813 |
| img2.jpg | 0.5362 | 0.5540 | 0.5340 | 0.5521 |

Discussão completa, escolha do fator de limiar e demais valores de
sigma estão em [`relatorio_gradiente.pdf`](relatorio_gradiente.pdf).

## Como executar

### Pré-requisitos

```bash
pip install opencv-python numpy scikit-image
```

### Rodar

```bash
python gradiente.py
```

O script processa as 4 imagens de teste (`moedas.png`,
`Lua1_gray.jpg`, `chessboard_inv.png`, `img2.jpg`) com os operadores
Prewitt e Scharr, imprime no terminal o limiar adaptativo e o SSIM de
cada combinação, e salva as imagens de bordas resultantes
(`resultado_<operador>_<método>_<imagem>`) e as referências do Canny
(`resultado_canny_sigma<σ>_<imagem>`).

## Estrutura do projeto

```
├── gradiente.py             -> implementação (convolução, gradiente, NMS, limiar, avaliação)
├── relatorio_gradiente.pdf   -> relatório com metodologia e análise de resultados
├── moedas.png                 -> imagens de teste
├── Lua1_gray.jpg
├── chessboard_inv.png
├── img2.jpg
└── resultado_*                -> saídas geradas pelo script
```

## Tecnologias

- Python 3
- OpenCV (`opencv-python`) — leitura/conversão de imagens e filtro Gaussiano
- NumPy — operações matriciais
- scikit-image — detector de Canny e cálculo de SSIM

## Equipe

- Integrantes: Karla Bertol & Vitor Cardoso Burgarelli
