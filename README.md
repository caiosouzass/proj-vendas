# Case Técnico – Especialista de Dados I

Análise exploratória, modelo de previsão de vendas diárias e proposta de arquitetura de IA para análise de imagens.

## Estrutura

```
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb   # Seção 1 – insights de negócio
│   └── 02_modelagem_preditiva.ipynb    # Seção 1 – previsão de receita e pedidos
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
   cd .. && python presentation/build_deck.py
   ```

## Principais conclusões

- **Qualidade dos dados:** 6% das linhas têm categoria corrompida (mantidas, rotuladas como `NÃO IDENTIFICADA`);
  ~5% das linhas têm receita ≤ 0 (−4,3% da receita líquida), compatível com estornos/cancelamentos.
- **Comportamento de vendas:** a intensidade promocional é o principal motor da receita diária. A Black Friday (28/11/2025)
  vale ~7,5× a mediana diária. Quarta a sexta são os dias mais fortes e domingo o mais fraco; a receita se concentra entre 10h e 22h.
- **Modelo:** previsão com 7 dias de antecedência usando Ridge, LightGBM e a média dos dois, validada por *walk-forward* e *holdout* (junho/2026).
  Os modelos reduzem o erro frente ao baseline sazonal, mas o ganho é moderado porque as campanhas não estão no calendário.
  O cenário com a taxa de desconto do dia é um teto otimista (circular); um teste de sensibilidade mostra que, com a taxa defasada, não há ganho sobre o cenário só com calendário e histórico.
  Detalhes e limites estão no notebook 02.
- **Seção 2:** proposta de pipeline com embeddings + score supervisionado, MVP em 6–8 semanas e critério go/no-go.

## Premissas

- `vlr_venda_desconto` é interpretado como desconto concedido (venda bruta ≈ receita + desconto).
- Linhas com receita ≤ 0 foram mantidas por comporem a receita líquida.
