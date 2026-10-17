# Backlog Técnico — Backend (Macro Aba 1)

## B1 detalhado (Backend) — Infra + validação global

## Objetivo do B1
Padronizar entrada de imagem e erros em **todos os endpoints já existentes**, sem mudar comportamento funcional dos algoritmos.

Endpoints cobertos no B1:
- `POST /api/v1/image-processing/grayscale`
- `POST /api/v1/image-processing/blur/gaussian`
- `POST /api/v1/image-processing/edges/sobel`
- `POST /api/v1/image-processing/edges/canny`

Base atual:
- Router: `backend/app/api/image_processing.py`
- Serviços: `backend/app/services/image_processing/filters.py`
- Testes: `backend/tests/test_image_processing.py`

---

## Checklist técnico executável (ordem recomendada)

## Tarefa 1 — Centralizar constantes de validação
- [x] Criar `backend/app/services/image_processing/constraints.py`
- [x] Definir constantes:
  - `MIN_IMAGE_SIZE = 50`
  - `MAX_IMAGE_SIZE = 200`
  - `ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}`

### Definição de pronto
- [x] Nenhum endpoint mantém “números mágicos” de tamanho/tipo.

## Tarefa 2 — Centralizar decode/encode/validação de imagem
- [x] Criar `backend/app/services/image_processing/io_utils.py`
- [x] Implementar funções:
  - `decode_image(image_bytes: bytes) -> np.ndarray`
  - `encode_png(image: np.ndarray) -> bytes`
  - `validate_dimensions(image: np.ndarray, min_size: int = 50, max_size: int = 200) -> None`
  - `count_pixels(image: np.ndarray) -> int` (já prepara Fundamentos)
- [x] Em `decode_image`, lançar exceção de domínio se bytes inválidos.
- [x] Em `validate_dimensions`, validar altura e largura (ambas no intervalo).

### Definição de pronto
- [x] `filters.py` não usa mais decode/encode duplicado.
- [x] Todas as validações de tamanho passam por uma única função.

## Tarefa 3 — Padronizar exceções de domínio
- [x] Criar `backend/app/services/image_processing/errors.py`
- [x] Definir exceções:
  - `InvalidImageError`
  - `UnsupportedContentTypeError`
  - `EmptyFileError`
  - `InvalidImageSizeError`
- [x] Mensagens estáveis (para teste):
  - `"Unsupported file type. Use PNG or JPEG/JPG."`
  - `"Empty file uploaded"`
  - `"Invalid or unsupported image file"`
  - `"Image dimensions must be between 50x50 and 200x200 pixels"`

### Definição de pronto
- [x] Sem `ValueError` genérico vazando para API.
- [x] Mensagens de erro consistentes entre endpoints.

## Tarefa 4 — Refatorar serviços existentes para reutilizar utilitários
- [x] Editar `backend/app/services/image_processing/filters.py`
- [x] Cada função (`grayscale_from_bytes`, `gaussian_blur_from_bytes`, `sobel_edges_from_bytes`, `canny_edges_from_bytes`) deve:
  1. Decodificar via `decode_image`
  2. Validar dimensão via `validate_dimensions`
  3. Processar
  4. Encodar via `encode_png`

### Definição de pronto
- [x] Repetição de `cv2.imdecode`/`cv2.imencode` removida.
- [x] Todos os algoritmos existentes continuam respondendo PNG.

## Tarefa 5 — Padronizar camada de API (router)
- [x] Editar `backend/app/api/image_processing.py`
- [x] Criar helper único para upload:
  - valida content-type,
  - lê bytes,
  - valida arquivo vazio,
  - delega ao serviço.
- [x] Mapear exceções de domínio para `HTTPException(400, detail=...)`.
- [x] Manter validações específicas já existentes:
  - gaussian: kernel ímpar > 1
  - sobel: kernel ímpar e {1,3,5,7}
  - canny: `threshold1 < threshold2`

### Definição de pronto
- [x] Fluxo de erro igual em todos endpoints.
- [x] Nenhuma mudança de contrato de URL/query atual.

## Tarefa 6 — Testes exatos (novos e ajustes)

Editar `backend/tests/test_image_processing.py` (ou quebrar em `test_image_validation.py`).

### Helpers de teste
- [x] Atualizar helper para suportar tamanho variável:
  - `_build_test_png(width: int, height: int) -> bytes`

### Casos obrigatórios (exatos)

#### A) Validação global de tipo/arquivo
- [x] `test_all_endpoints_reject_unsupported_content_type` (parametrizado 4 endpoints)
- [x] `test_all_endpoints_reject_empty_file` (parametrizado 4 endpoints)
- [x] `test_all_endpoints_reject_invalid_image_bytes_with_image_content_type` (parametrizado 4 endpoints)

#### B) Validação global de dimensão
- [x] `test_all_endpoints_reject_image_smaller_than_min_50x50` (ex.: 49x50)
- [x] `test_all_endpoints_reject_image_larger_than_max_200x200` (ex.: 201x200)
- [x] `test_all_endpoints_accept_image_exactly_50x50`
- [x] `test_all_endpoints_accept_image_exactly_200x200`

#### C) Validações específicas já existentes
- [x] gaussian: kernel par rejeita
- [x] sobel: kernel fora do conjunto rejeita
- [x] canny: threshold1 >= threshold2 rejeita

### Definição de pronto
- [x] Suite de testes passa com cobertura dos cenários transversais.
- [x] Mensagens de erro esperadas estão assertadas.

## Tarefa 7 — Documentação mínima sincronizada
Editar `README.md`:
- [x] Adicionar regra global de tamanho (50x50 até 200x200).
- [x] Declarar que vale para **todos** endpoints de image processing.
- [x] Manter lista de endpoints atuais.

### Definição de pronto
- [x] README reflete comportamento real do backend.

## Comandos de validação (exatos)

No diretório raiz:
1. `cd backend`
2. `python -m pytest -q`

Critério de aceite B1:
- [x] Todos testes verdes.
- [x] Endpoints existentes funcionando com mesmo contrato.
- [x] Validação global ativa para os 4 endpoints.

---

## Backlog técnico executável (próximas etapas por aba)

## Aba 1 — Fundamentos (backend)
- [ ] Endpoint `POST /api/v1/image-processing/fundamentals/metadata`
  - retorna `width`, `height`, `channels`, `total_pixels`.
- [ ] Endpoint `POST /api/v1/image-processing/fundamentals/kernel-window`
  - params: `kernel_size` (3|4|5), `x`, `y`, `mode` (`kernel`|`full`).
- [ ] Testes de borda (coordenadas inválidas e kernel fora da imagem).

### DoD
- [ ] Metadata correta e estável.
- [ ] Janela/kernel retornada corretamente para 3x3, 4x4 e 5x5.

## Aba 2 — Canais de cores (backend)
- [ ] Conversões RGB→Gray/HSV/HSL/HSI/YCbCr/CMYK.
- [ ] Extração de canal por `channel_id`.
- [ ] Testes de shape/faixa por conversão e extração.

### DoD
- [ ] Cada operação retorna imagem válida PNG.
- [ ] `channel_id` inválido retorna 400 com mensagem clara.

## Aba 3 — Filtros (backend)
- [ ] Lineares + não lineares + especializados fase 1.
- [ ] Padronizar parâmetro `kernel_size` e validações.
- [ ] Testes parametrizados por filtro.

### DoD
- [ ] Todos filtros listados da fase retornam PNG e validam params.

## Aba 4 — Binarização
- [ ] threshold global variants + Otsu/Triangle + adaptativos iniciais.
- [ ] Testes de binário (apenas 0/255 quando aplicável).

### DoD
- [ ] Operações disponíveis e com saída coerente por método.

## Aba 5 — Histograma
- [ ] Endpoints JSON para histogramas.
- [ ] Endpoints imagem para transformações.
- [ ] Testes de soma/normalização/acumulado.

### DoD
- [ ] Pipeline “imagem → hist → transformação → novo hist” suportado no backend.

## Aba 6 — Morfologia
- [ ] erosão/dilatação/abertura/fechamento + avançados fase 1.
- [ ] Testes com imagens sintéticas simples.

### DoD
- [ ] Operações morfológicas básicas estáveis com kernel parametrizável.

## Aba 7 — Transformações geométricas
- [ ] translação/rotação/escala/reflexão/shear + affine/perspective.
- [ ] interpolação: nearest/bilinear/bicubic/lanczos.

### DoD
- [ ] Transformações preservam contrato e validam parâmetros.

## Aba 8 — Detecção de bordas
- [ ] clássicos + canny/log/dog/zero-crossing.
- [ ] teste de regressão visual simples (shape/tipo/faixa).

### DoD
- [ ] Saída de bordas consistente e parametrização documentada.

## Aba 9 — Contornos
- [ ] detecção + aproximação + métricas (área, perímetro, circularidade etc).
- [ ] retorno JSON estruturado.

### DoD
- [ ] Métricas retornadas com nomes e unidades consistentes.

## Aba 10 — Segmentação
- [ ] clássica fase 1 (threshold family/watershed/kmeans/grabcut).
- [ ] avançada fase 2 (snakes/level set etc., se dependências permitirem).

### DoD
- [ ] Segmentações principais disponíveis com testes de sanidade.
