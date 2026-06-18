# Relatório – Trabalho Prático: Operador Gradiente na Detecção de Bordas

## 1. Identificação da equipe

- Nomes: Karla Bertol & Vitor Cardoso Burgarelli
- Disciplina: PIM
- Professor: Gilmário

## 2. Identificação da tarefa

O trabalho consiste na implementação do operador gradiente para detecção de
bordas em imagens em tons de cinza, utilizando os operadores de Prewitt e
Scharr. A convolução entre os kernels e a imagem foi implementada
manualmente (sem uso de funções prontas de convolução). Após o cálculo da
magnitude e direção do gradiente, foi aplicada a etapa de seleção dos
gradientes mais significativos (supressão de não-máximos), utilizando dois
métodos:

- **b.1**: comparação com os dois vizinhos mais próximos, escolhidos por
  octante da direção do gradiente;
- **b.2**: comparação com vizinhos obtidos por interpolação bilinear,
  conforme a proporção entre as componentes Gx e Gy.

Por fim, os resultados foram comparados com o detector de bordas de Canny
(`skimage.feature.canny`) por meio do índice **SSIM** (Structural Similarity
Index).

## 3. Explicação conceitual sobre a solução fornecida

### 3.1 Entrada

- Imagens de teste: `moedas.png`, `Lua1_gray.jpg`, `chessboard_inv.png`,
  `img02.jpg`.
- Imagens convertidas para tons de cinza (caso estejam em RGB).

### 3.2 Processo (pipeline)

1. **Pré-filtragem**: aplicação de filtro Gaussiano 3x3 (`cv2.GaussianBlur`)
   para atenuar ruído antes da diferenciação, já que filtros derivativos são
   sensíveis a altas frequências.
2. **Cálculo do gradiente (Gx, Gy)**: convolução manual da imagem suavizada
   com os kernels de Prewitt ou Scharr.
3. **Cálculo da magnitude e direção**:
   - `M(i,j) = sqrt(Gx² + Gy²)`
   - `θ(i,j) = atan2(Gy, Gx)`, mapeado para o intervalo `[0°, 180°)`
4. **Seleção dos gradientes mais significativos (supressão de não-máximos)**:
   - **b.1**: o pixel é mantido como borda se sua magnitude for maior que a
     dos dois vizinhos colineares na direção do gradiente, escolhidos entre
     os 8 vizinhos mais próximos conforme o octante de θ.
   - **b.2**: ao invés de usar diretamente os vizinhos discretos, os valores
     comparados são interpolados linearmente entre dois pixels adjacentes,
     usando como peso a razão `t = |componente menor| / |componente maior|`
     entre Gx e Gy. Isso produz uma estimativa mais precisa do valor do
     gradiente exatamente na direção θ.
5. **Binarização**: aplicação de limiar fixo (`limiar_borda = 50`) sobre a
   imagem resultante da supressão, gerando a imagem binária final de bordas.
6. **Comparação com Canny**: aplicação de `skimage.feature.canny` (sigma=1.0)
   sobre a imagem suavizada e cálculo do índice SSIM entre essa saída e a
   imagem binária produzida pela implementação própria.

### 3.3 Saída

Para cada imagem de teste e cada operador (Prewitt e Scharr), foram geradas:

- Imagem de bordas pelo método b.1 (`resultado_<operador>_b1_<imagem>`)
- Imagem de bordas pelo método b.2 (`resultado_<operador>_b2_<imagem>`)
- Imagem de bordas pelo Canny (`resultado_canny_<operador>_<imagem>`)

Total de imagens geradas: **4 imagens de teste × 2 operadores × (2 métodos de
supressão + 1 Canny) = 24 imagens** (sendo 16 referentes apenas às saídas
b1/b2 da implementação própria — 4 imagens × 2 operadores × 2 métodos).

> _Sugestão: incluir aqui um print/colagem comparando lado a lado: imagem
> original, b1, b2 e Canny, para pelo menos uma imagem (ex.: moedas.png)._

### 3.4 Diagrama do pipeline

```
Imagem RGB/cinza
      |
      v
Filtro Gaussiano (redução de ruído)
      |
      v
Convolução manual (Gx, Gy) -- Prewitt ou Scharr
      |
      v
Magnitude M(i,j) e Direção θ(i,j)
      |
      v
Supressão de não-máximos (b.1 ou b.2)
      |
      v
Binarização (limiar = 50)
      |
      v
Imagem de bordas final
      |
      v
Comparação com Canny (SSIM)
```

## 4. Instruções sobre a "compilação"

- Linguagem: Python 3.
- Execução via linha de comando (terminal Fedora):
  ```
  python gradiente.py
  ```
- Não foi utilizada IDE específica para execução, apenas editor de texto +
  terminal.
- Bibliotecas utilizadas:
  - `opencv-python` (cv2) – leitura, conversão de cor, filtro Gaussiano,
    gravação de imagens.
  - `numpy` – operações matriciais (magnitude, direção, vetorização).
  - `scikit-image` (skimage.feature, skimage.metrics) – detector de Canny e
    cálculo do SSIM.
  - `math` – funções trigonométricas auxiliares.
- As imagens de entrada (`moedas.png`, `Lua1_gray.jpg`, `chessboard_inv.png`,
  `img02.jpg`) devem estar no mesmo diretório do script.

## 5. Teste de mesa

> _Opcional, mas recomendado para reforçar o entendimento do método b.2._

Exemplo de teste de mesa para um pixel com `Gx = 4` e `Gy = 2` (octante 1,
0° ≤ θ ≤ 45°):

- `t = |Gy| / |Gx| = 2 / 4 = 0.5`
- `vizinho_Y = (1 - 0.5) * F + 0.5 * C`
- `vizinho_X = (1 - 0.5) * D + 0.5 * G`
- Se `M(i,j) > vizinho_Y` e `M(i,j) > vizinho_X`, o pixel é considerado borda.

Isso mostra que quando `Gx = Gy` (θ = 45°), `t = 1` e o vizinho interpolado
coincide totalmente com o vizinho diagonal (C ou G), enquanto quando `Gy = 0`
(θ = 0°), `t = 0` e o vizinho coincide totalmente com o vizinho horizontal
(F ou D) — comportamento coerente com o esperado.

## 6. Análise de resultados

### 6.1 Tabela de SSIM (implementação própria vs. Canny)

| Imagem              | Prewitt b1 | Prewitt b2 | Scharr b1 | Scharr b2 |
|---------------------|-----------:|-----------:|----------:|----------:|
| moedas.png          | 0.6788     | 0.7138     | 0.6225    | 0.6600    |
| Lua1_gray.jpg       | 0.8926     | 0.8992     | 0.7580    | 0.7627    |
| chessboard_inv.png  | 0.8107     | 0.8107     | 0.8125    | 0.8126    |
| img02.jpg           | 0.8485     | 0.8651     | 0.2645    | 0.2764    |

### 6.2 Discussão

- **b.2 (interpolação) sempre apresentou SSIM maior ou igual a b.1**, o que é
  esperado: a interpolação fornece uma estimativa mais precisa do gradiente
  na direção exata de θ, reduzindo falsos negativos/positivos na supressão de
  não-máximos e aproximando o resultado do comportamento "ideal" capturado
  pelo Canny.

- **chessboard_inv.png** apresentou praticamente o mesmo SSIM em b1 e b2 para
  ambos os operadores. Isso ocorre porque as bordas dessa imagem são
  predominantemente horizontais e verticais (θ ≈ 0° ou 90°), casos em que
  `t ≈ 0`, fazendo com que o vizinho interpolado coincida com o vizinho
  discreto já usado em b.1. Ou seja, a interpolação "degenera" para o caso
  b.1 quando as bordas são alinhadas aos eixos.

- **img02.jpg com Scharr apresentou queda brusca no SSIM (≈0.26)**, muito
  abaixo dos demais resultados. A explicação mais provável está relacionada
  à maior sensibilidade do operador Scharr: seus pesos centrais são bem
  maiores (10 e 3) que os de Prewitt (1), tornando-o muito mais sensível a
  variações de intensidade sutis na imagem. Como resultado, a magnitude do
  gradiente em `img02.jpg` fica elevada em uma quantidade muito maior de
  pixels, e o limiar fixo de binarização (`limiar_borda = 50`) não é
  suficiente para descartar essas bordas "fracas". A imagem binarizada
  resultante fica, portanto, muito mais densa de bordas do que a saída do
  Canny (que possui seu próprio mecanismo de limiar adaptativo/histerese),
  reduzindo drasticamente a similaridade estrutural entre as duas.

- **De forma geral, o operador Prewitt apresentou SSIM mais próximo do Canny
  do que o Scharr** em quase todas as imagens, sugerindo que, para o limiar
  fixo escolhido, Prewitt produz resultados mais "comparáveis" ao Canny,
  enquanto Scharr exigiria um limiar mais alto ou uma normalização adicional
  da magnitude para se tornar comparável.

### 6.3 Possíveis melhorias (trabalhos futuros)

- Uso de limiar adaptativo (ex.: baseado na mediana da magnitude, conforme
  sugerido no método b.3 de seleção por histerese).
- Normalização da magnitude do gradiente antes da binarização, especialmente
  para o operador Scharr.
- Aplicação de histerese de dois limiares (semelhante ao Canny) para conectar
  fragmentos de borda.
