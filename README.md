# Case Técnico – Especialista de Dados I

Análise exploratória, modelo de previsão de vendas diárias e proposta de arquitetura de IA para análise de imagens.

## Estrutura

```
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb   # Seção 1 – insights de negócio
│   ├── 02_modelagem_preditiva.ipynb    # Seção 1 – indicadores, features, modelos, ajuste e importância
│   └── 03_comparativo_modelos.ipynb    # Seção 1 – comparação com ETS, SARIMAX e Prophet
├── src/                                # Código reutilizável (dados, features, modelagem, estilo dos gráficos)
├── docs/secao2_arquitetura_ia.md       # Seção 2 – inovação e arquitetura de IA
├── presentation/                       # Apresentação executiva (.pptx) e script que a gera
├── reports/                            # Figuras e números-chave gerados pelos notebooks
└── requirements.txt
```

## Como reproduzir

1. Criar o ambiente e instalar dependências:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Colocar a base em `data/raw/vendas.csv` (a pasta `data/` não é versionada).
3. Executar os notebooks, em ordem (a partir da pasta `notebooks/`):
   ```bash
   cd notebooks
   jupyter nbconvert --to notebook --execute --inplace 01_analise_exploratoria.ipynb
   jupyter nbconvert --to notebook --execute --inplace 02_modelagem_preditiva.ipynb
   jupyter nbconvert --to notebook --execute --inplace 03_comparativo_modelos.ipynb   # ~10 min (reajusta os modelos clássicos a cada dia de teste)
   cd .. && python presentation/build_deck.py
   ```

## Principais conclusões

- **Qualidade dos dados:** 6% das linhas têm categoria corrompida (mantidas, rotuladas como `NÃO IDENTIFICADA`);
  ~5% das linhas têm receita ≤ 0 (−4,3% da receita líquida). Os testes indicam que não são estornos (mesma distribuição de valor por item das vendas, sem relação com vendas anteriores); o padrão sugere inversão de sinal aleatória, a confirmar com o dono do dado. Mantidas como vieram, com análise de sensibilidade no notebook 02 (manter × descartar × |receita|): o erro relativo do modelo praticamente não muda, mas o nível sim (descartar: receita +4,3% e pedidos −4,1%; |receita|: receita +8,6%).
- **Comportamento de vendas:** a venda se concentra em eventos (Beauty Week e Black Friday) e acompanha, de forma moderada, a intensidade promocional (correlação de 0,55 com a taxa de desconto, que cai para 0,26 sem novembro). Quarta a sexta são os dias mais fortes e domingo o mais fraco; a receita se concentra entre 10h e 22h.
- **Indicadores:** projetamos receita aprovada e nº de pedidos (total diário). Itens vendidos andam junto com pedidos (correlação ~1,0), o desconto é insumo (decisão do comercial) e ticket e preço por item são derivados.
- **Sazonalidade:** novembro (Beauty Week) concentra ~26% da receita e ~34% dos pedidos do período; o dia da Black Friday vale ~2,6% e 7,5× a mediana diária. O modelo tem uma variável para o mês inteiro e outras para o pico; como há um único novembro, o efeito não pode ser validado fora da amostra.
- **Modelo:** projeção com 7 dias de antecedência usando Ridge, LightGBM e a média dos dois, validada por *walk-forward* (4 janelas) e *holdout* (junho/2026). O WAPE da receita fica em ~24% contra ~34% do baseline sazonal (pedidos: ~27% contra ~38%).
  O cenário com a taxa de desconto do dia é um teto otimista (circular); com a taxa defasada não há ganho sobre o cenário só com calendário e histórico.
- **Ajuste e convergência (notebook 02):** busca em grade com validação temporal aninhada. Só o Ridge melhora de forma consistente (α = 300); nas árvores a melhor configuração das janelas internas não generaliza, e a curva de convergência mostra sobreajuste após ~70 árvores.
- **Comparativo (notebook 03):** Ridge, SARIMAX com eventos e a média empatam no walk-forward (≤ 0,5 p.p.); XGBoost é pior no walk-forward e melhor no holdout (inconsistente); Prophet e ETS ficam atrás. Nenhum modelo passa de ~24% de WAPE na receita: o limite vem da informação disponível, não do algoritmo.
- **Importância das variáveis (fora da amostra):** pesam o dia da semana, o Dia das Mães e a tendência; o histórico recente e a taxa de desconto não ajudam de forma estável. Com 56 dias de teste, o ranking é um indício.
- **Seção 2:** pipeline com embeddings + score supervisionado; caminho enquanto a ferramenta não é homologada (protótipo em Python com imagens públicas e produtização depois com a área de Tecnologia); MVP em 6–8 semanas e critério go/no-go.

## Premissas

- `vlr_venda_desconto` é interpretado como desconto concedido (venda bruta ≈ receita + desconto).
- Linhas com receita ≤ 0 foram mantidas como vieram (base líquida); a leitura alternativa (|receita|) é avaliada nos notebooks.
