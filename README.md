# Case Técnico – Especialista de Dados I

Análise exploratória, modelo de previsão de vendas diárias e proposta de arquitetura de IA para análise de imagens.

## Estrutura

```
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb   # Seção 1 – insights de negócio
│   ├── 02_modelagem_preditiva.ipynb    # Seção 1 – previsão de receita e pedidos
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
   jupyter nbconvert --to notebook --execute --inplace 03_comparativo_modelos.ipynb   # ~8 min (reajusta os modelos clássicos a cada dia de teste)
   cd .. && python presentation/build_deck.py
   ```

## Principais conclusões

- **Qualidade dos dados:** 6% das linhas têm categoria corrompida (mantidas, rotuladas como `NÃO IDENTIFICADA`);
  ~5% das linhas têm receita ≤ 0 (−4,3% da receita líquida). Os testes indicam que não são estornos (mesma distribuição de valor por item das vendas, sem relação com vendas anteriores); o padrão sugere inversão de sinal aleatória, a confirmar com o dono do dado. Mantidas como vieram, com análise de sensibilidade (|receita|: +8,6% no nível, erro do modelo praticamente igual).
- **Comportamento de vendas:** a intensidade promocional é o principal motor da receita diária. A Black Friday (28/11/2025)
  vale ~7,5× a mediana diária. Quarta a sexta são os dias mais fortes e domingo o mais fraco; a receita se concentra entre 10h e 22h.
- **Modelo:** previsão com 7 dias de antecedência usando Ridge, LightGBM e a média dos dois, validada por *walk-forward* e *holdout* (junho/2026).
  Os modelos reduzem o erro frente ao baseline sazonal, mas o ganho é moderado porque as campanhas não estão no calendário.
  O cenário com a taxa de desconto do dia é um teto otimista (circular); um teste de sensibilidade mostra que, com a taxa defasada, não há ganho sobre o cenário só com calendário e histórico.
  Detalhes e limites estão no notebook 02.
- **Comparativo (notebook 03):** o SARIMAX com variáveis de eventos empata com a média Ridge+LightGBM (diferença de ~1 p.p. de WAPE, dentro do ruído); Prophet e ETS ficam atrás. Nenhum modelo passa de ~24–25% de WAPE na receita: o limite vem da informação disponível, não do algoritmo.
- **Importância das variáveis (fora da amostra):** pesam o histórico do mesmo dia da semana, o dia da semana, a tendência e o Dia das Mães; a taxa de desconto não ajuda a prever dias novos.
- **Seção 2:** proposta de pipeline com embeddings + score supervisionado, MVP em 6–8 semanas e critério go/no-go.

## Premissas

- `vlr_venda_desconto` é interpretado como desconto concedido (venda bruta ≈ receita + desconto).
- Linhas com receita ≤ 0 foram mantidas como vieram (base líquida); a leitura alternativa (|receita|) é avaliada nos notebooks.
