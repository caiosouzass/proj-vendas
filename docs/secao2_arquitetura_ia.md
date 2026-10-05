# Seção 2 – Inovação e Arquitetura de IA (Business Case)

**Situação:** um time do hackathon quer um pipeline que classifique e pontue (*scoring*) as imagens/banners do site
por características qualitativas e quantitativas. O time não programa e a ferramenta *no-code* escolhida não é homologada no grupo.

## 1. Como conduzir a conversa com o time

1. **Reconhecer o valor da ideia antes do bloqueio.** A iniciativa nasce de cultura de inovação; o "não" à ferramenta
   precisa vir acompanhado de um "sim, e o caminho é este".
2. **Explicar o porquê da homologação em linguagem de negócio:** imagens e dados de comportamento do site passam por
   fornecedores externos; sem homologação há risco de vazamento, descumprimento de LGPD, custos fora de controle e
   solução que ninguém sabe sustentar.
3. **Transformar a ideia em hipótese mensurável.** Em um workshop curto (2h) com o time, definir:
   - qual decisão o score vai apoiar (escolher banner da home, priorizar testes A/B, orientar o time de design)?
   - o que é um "banner bom" (clique, conversão, receita da sessão) – esse é o alvo que dá sentido ao score;
   - quem usa o score e com que frequência.
4. **Co-criar, não terceirizar.** O time de negócio mantém a titularidade da pergunta e dos critérios qualitativos; Dados cuida
   da implementação. Entrega em ciclos curtos (demo quinzenal) para manter engajamento.
5. **Alternativa imediata sem código:** enquanto o MVP não existe, usar ferramentas já homologadas (planilhas, BI, o
   processo de teste A/B atual) para rotular manualmente 100–200 banners. Esses rótulos serão o gabarito do modelo.

## 2. Implementação técnica viável (sem restrição de ferramenta)

**Princípio:** representar cada imagem por *embeddings* (vetores que resumem o conteúdo visual) e por métricas visuais
clássicas, e treinar um modelo leve que aprende a relação entre essas características e o desempenho real do banner.

```
Banners (CMS/site) ─► Ingestão e versionamento ─► Extração de características ─► Score ─► Consumo
                                                   │                              │
                                                   ├─ Embeddings (CLIP/SigLIP)    ├─ Modelo supervisionado
                                                   ├─ Métricas quantitativas      │   (GBM/regressão) no alvo CTR/conversão
                                                   └─ Atributos qualitativos      └─ Explicação (SHAP) por banner
                                                      (modelo multimodal)
                                                                      Consumo: dashboard de BI, API, relatório semanal
```

| Camada | O que faz | Opções |
|---|---|---|
| Quantitativo | brilho, contraste, saturação, paleta dominante, proporção de texto, densidade de elementos, saliência | OpenCV, scikit-image |
| Embeddings | vetor semântico da imagem; permite similaridade e agrupamento de estilos | CLIP/SigLIP (open source, rodam em CPU/GPU pequena) |
| Qualitativo | presença de produto/modelo, legibilidade do CTA, tom (promoção × institucional) | modelo multimodal via API homologada ou classificador *zero-shot* sobre os embeddings |
| Score | combina as camadas e prevê desempenho (CTR/conversão) | LightGBM ou regressão regularizada; poucos milhares de banners bastam |
| Armazenamento | vetores e metadados para busca e histórico | Postgres + pgvector (ou o armazém de dados do grupo) |
| Consumo | visualização, ranking, alertas | dashboard no BI corporativo; API para o CMS |

**Pontos de cuidado técnico:**
- Validar o score contra desempenho **observado** (correlação e *lift* por faixa de score), com separação temporal; sem isso é só opinião automatizada.
- Controlar efeitos de contexto (posição do banner, campanha, sazonalidade), senão o modelo aprende "banner da Black Friday vende mais".
- Manter humano no ciclo: o score orienta, a decisão final é do time de design/negócio.
- Monitorar deriva (novos estilos de banner mudam a distribuição dos embeddings) e reavaliar periodicamente.

## 3. Como justificar para a gestão sênior: MVP e custo × retorno

**Proposta de MVP (6–8 semanas, 1 cientista de dados + 1 engenheiro em tempo parcial + 1 representante do negócio):**

| Semana | Entrega |
|---|---|
| 1–2 | Levantamento de banners históricos e métricas; definição do alvo; rotulagem de uma amostra |
| 3–4 | Extração de características e embeddings; primeiro modelo e análise de poder preditivo |
| 5–6 | Score explicável por banner; dashboard simples; teste cego com o time de design |
| 7–8 | Piloto: usar o score para priorizar 1 ciclo de testes A/B; medir efeito |

**Critério de continuidade (decisão go/no-go ao fim do MVP):** o score separar banners de alto e baixo desempenho
de forma estatisticamente consistente em dados fora da amostra e o time de design aprovar a utilidade na prática.

**Custo.** Predominantemente tempo da equipe. A infraestrutura é pequena: embeddings de alguns milhares de imagens
rodam em horas numa máquina modesta, e o uso de API multimodal (se escolhida) escala com o número de imagens, que aqui é
baixo. Os custos recorrentes (nuvem, armazenamento, API) devem ser estimados com a cotação vigente do fornecedor
homologado antes do piloto.

**Retorno.** Três alavancas, a quantificar com dados do próprio site:
1. **Menos tentativa e erro:** reduzir o número de testes A/B necessários para encontrar um banner vencedor.
2. **Mais conversão:** pequeno ganho de CTR/conversão em espaços de alto tráfego se multiplica pelo volume.
3. **Padrão de criação:** conhecimento reutilizável para o time de design (o que funciona em cada categoria/campanha).

**Racional de break-even (a preencher com números reais no início do projeto):**

```
Ganho mensal = receita mensal atribuível aos banners × lift esperado em conversão
Break-even    = custo do MVP + custo mensal de operação ≤ Ganho acumulado
```

Em vez de prometer um *lift*, a proposta pede a verba do MVP para **medir** o lift num piloto controlado –
o investimento é baixo e a decisão seguinte (escalar ou encerrar) usa evidência.

**Riscos e mitigação:** score sem correlação com desempenho (mitigado pelo critério go/no-go), dependência de rótulos
subjetivos (mitigado pelo uso de desempenho observado como alvo), adoção baixa pelo time (mitigada pela co-criação) e
questões de privacidade/segurança (mitigadas por usar apenas ferramentas homologadas e imagens institucionais, sem dados pessoais).
