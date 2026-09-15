#!/usr/bin/env python3
"""Build report/evaluation_answers.html — the three evaluation questions, answered
with the figures from report/figures/answers/ (embedded as data URIs, self-contained).

  python report/make_answer_page.py
"""
from base64 import b64encode
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "report/figures/answers"
OUT = ROOT / "report/evaluation_answers.html"


def img(name, alt):
    data = b64encode((FIGDIR / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def plate(n, name, title, alt, caption):
    return f"""<figure class="plate">
  <figcaption class="plate-head"><span class="fignum">Fig. {n}</span>{title}</figcaption>
  <div class="plate-paper">{img(name, alt)}</div>
  <p class="plate-cap">{caption}</p>
</figure>"""


HEAD = """<title>Métriques MT à l'épreuve de la longueur</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  color-scheme: light;
  --bg: #f7f7f4;
  --surface: #ffffff;
  --surface-2: #f1f1ec;
  --ink: #16181c;
  --ink-2: #4a4f58;
  --muted: #767b84;
  --rule: #e0e0da;
  --rule-strong: #c9cac2;
  --accent: #1f66be;
  --accent-soft: #e8f0fb;
  --warm: #c2521f;
  --warm-soft: #fbeee7;
  --flag: #8a6100;
  --flag-soft: #f8f0dc;
  --paper: #fcfcfb;
  --paper-rule: #dcdcd5;
  --shadow: 0 1px 2px rgba(20, 22, 26, .05), 0 8px 24px -18px rgba(20, 22, 26, .35);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121417;
    --surface: #191c21;
    --surface-2: #21252b;
    --ink: #edeef0;
    --ink-2: #b9bec6;
    --muted: #8d949d;
    --rule: #2b3037;
    --rule-strong: #3b424b;
    --accent: #74a9ee;
    --accent-soft: #17273c;
    --warm: #ef8b5f;
    --warm-soft: #33211a;
    --flag: #d6a94a;
    --flag-soft: #2c2517;
    --paper: #fcfcfb;
    --paper-rule: #33383f;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 10px 30px -20px rgba(0, 0, 0, .9);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121417;
  --surface: #191c21;
  --surface-2: #21252b;
  --ink: #edeef0;
  --ink-2: #b9bec6;
  --muted: #8d949d;
  --rule: #2b3037;
  --rule-strong: #3b424b;
  --accent: #74a9ee;
  --accent-soft: #17273c;
  --warm: #ef8b5f;
  --warm-soft: #33211a;
  --flag: #d6a94a;
  --flag-soft: #2c2517;
  --paper: #fcfcfb;
  --paper-rule: #33383f;
  --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 10px 30px -20px rgba(0, 0, 0, .9);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
  font-size: 16.5px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif; text-wrap: balance; margin: 0; }
a { color: var(--accent); }
code, .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .86em; }

.wrap { max-width: 1220px; margin: 0 auto; padding: 0 clamp(18px, 4vw, 44px); }

/* ── masthead ───────────────────────────────────────────── */
.masthead { border-bottom: 1px solid var(--rule); background: var(--surface); }
.masthead .wrap { padding-block: clamp(38px, 6vw, 72px) clamp(28px, 4vw, 44px); }
.eyebrow {
  font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: .13em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 18px;
}
h1 { font-size: clamp(34px, 5.2vw, 54px); line-height: 1.08; font-weight: 600; letter-spacing: -.015em; max-width: 17ch; }
.standfirst { margin: 20px 0 0; font-size: clamp(17px, 1.6vw, 19.5px); color: var(--ink-2); max-width: 62ch; }
.meta-row {
  display: flex; flex-wrap: wrap; gap: 10px 26px; margin-top: 26px;
  font-family: "IBM Plex Mono", monospace; font-size: 12.5px; color: var(--muted);
}

/* ── answers up front ───────────────────────────────────── */
.answers { display: grid; gap: 16px; grid-template-columns: repeat(3, 1fr); margin: clamp(30px, 5vw, 52px) 0 0; }
@media (max-width: 900px) { .answers { grid-template-columns: 1fr; } }
.ans {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
  padding: 20px 22px 22px; box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 10px;
}
.ans-q { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .12em;
         text-transform: uppercase; color: var(--accent); }
.ans-head { font-family: "Source Serif 4", serif; font-size: 20px; font-weight: 600; line-height: 1.25; }
.ans p { margin: 0; font-size: 15px; color: var(--ink-2); }
.ans a { text-decoration: none; border-bottom: 1px solid currentColor; font-size: 14px; align-self: flex-start; margin-top: auto; padding-top: 6px; }

/* ── sections ───────────────────────────────────────────── */
section { padding-block: clamp(46px, 7vw, 86px); border-top: 1px solid var(--rule); }
.sec-head { display: flex; gap: 18px; align-items: baseline; margin-bottom: 8px; }
.sec-num { font-family: "IBM Plex Mono", monospace; font-size: 13px; color: var(--warm); letter-spacing: .1em; padding-top: 4px; }
h2 { font-size: clamp(25px, 3.4vw, 34px); font-weight: 600; letter-spacing: -.01em; max-width: 24ch; }
.sec-sub { color: var(--muted); margin: 4px 0 0 0; max-width: 68ch; font-size: 15.5px; }
.prose { max-width: 70ch; }
.prose p { margin: 1.05em 0; }
.prose h3 { font-size: 19px; font-weight: 600; margin: 2em 0 .4em; }
.lede { font-size: 18.5px; color: var(--ink); border-left: 3px solid var(--accent); padding-left: 18px; margin: 26px 0 0; }
.lede strong { font-weight: 600; }
ul.tight { margin: .9em 0; padding-left: 1.1em; }
ul.tight li { margin: .38em 0; }
.num { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; font-size: .92em; color: var(--ink); }

/* ── figure plates ──────────────────────────────────────── */
.plate { margin: 34px 0 0; }
.plate-head {
  display: flex; gap: 12px; align-items: baseline; font-family: "Source Serif 4", serif;
  font-size: 17px; font-weight: 600; color: var(--ink); margin-bottom: 10px;
}
.fignum {
  font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .1em;
  color: var(--warm); text-transform: uppercase; white-space: nowrap; padding-top: 3px;
}
.plate-paper {
  background: var(--paper); border: 1px solid var(--paper-rule); border-radius: 2px;
  padding: 14px; overflow-x: auto; box-shadow: var(--shadow);
}
.plate-paper img { display: block; width: 100%; min-width: 620px; height: auto; }
.plate-cap { margin: 12px 0 0; font-size: 14.5px; color: var(--ink-2); max-width: 78ch; }
.plate-cap b { color: var(--ink); font-weight: 600; }

/* ── boxes ──────────────────────────────────────────────── */
.box {
  border: 1px solid var(--rule); border-radius: 3px; background: var(--surface);
  padding: 18px 22px; margin: 30px 0 0; max-width: 78ch;
}
.box h4 {
  margin: 0 0 8px; font-family: "IBM Plex Mono", monospace; font-size: 11.5px;
  letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 500;
}
.box p, .box li { font-size: 14.8px; color: var(--ink-2); margin: .5em 0; }
.box.flag { background: var(--flag-soft); border-color: color-mix(in srgb, var(--flag) 35%, transparent); }
.box.flag h4 { color: var(--flag); }

/* ── tables ─────────────────────────────────────────────── */
.tablewrap { overflow-x: auto; margin: 26px 0 0; border: 1px solid var(--rule); border-radius: 3px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; }
th, td { text-align: left; padding: 9px 14px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
thead th { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .08em;
           text-transform: uppercase; color: var(--muted); font-weight: 500; }
tbody tr:last-child td { border-bottom: 0; }
td.n { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
td.win { color: var(--accent); font-weight: 600; }
.rowlab { color: var(--ink); }
.fam-da td.rowlab::before, .fam-qe td.rowlab::before {
  content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 9px; vertical-align: middle;
}
.fam-da td.rowlab::before { background: #2a78d6; }
.fam-qe td.rowlab::before { background: #eb6834; }

/* ── footer ─────────────────────────────────────────────── */
footer { border-top: 1px solid var(--rule); background: var(--surface); padding-block: 40px 56px; }
footer .wrap { display: grid; gap: 26px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
footer h4 { margin: 0 0 8px; font-family: "IBM Plex Mono", monospace; font-size: 11.5px;
            letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
footer p, footer li { font-size: 14px; color: var(--ink-2); margin: .35em 0; }
footer ul { margin: 0; padding-left: 1.05em; }
</style>"""


def build():
    p = []
    a = p.append

    a(HEAD)

    # ── masthead ────────────────────────────────────────────
    a("""<header class="masthead"><div class="wrap">
<p class="eyebrow">FDTEM · rapport de résultats · 25 août 2026</p>
<h1>Métriques MT à l'épreuve de la longueur</h1>
<p class="standfirst">Trois cadres d'évaluation, trois questions. Quelles métriques suivent le mieux
le jugement humain selon la taille de l'entrée&nbsp;? Que détectent-elles sur MetaDocEval&nbsp;? Et que
deviennent les encodeurs quand on concatène des phrases FLORES+&nbsp;?</p>
<div class="meta-row">
  <span>10 figures</span><span>2 métriques publiées + 8 bras ré-entraînés</span>
  <span>5 encodeurs</span><span>sources&nbsp;: <span class="mono">results/</span></span>
</div>
<div class="answers">
  <article class="ans">
    <span class="ans-q">Q1 · corrélation humaine</span>
    <p class="ans-head">COMET-DA domine partout&nbsp;; la longueur coûte un tiers du signal.</p>
    <p>Les cinq bras DA passent devant les cinq bras Kiwi dans les trois régimes. Meilleur sur phrases
    seules&nbsp;: <span class="num">DA · phrases, τ&nbsp;=&nbsp;.410</span>. Sur documents natifs&nbsp;:
    <span class="num">DA · texte long, τ&nbsp;=&nbsp;.288</span>, contre <span class="num">.397</span> sur phrases.</p>
    <a href="#q1">Voir les figures →</a>
  </article>
  <article class="ans">
    <span class="ans-q">Q2 · MetaDocEval</span>
    <p class="ans-head">Écart net DA / Kiwi sur le lexical, aucun gain de contexte.</p>
    <p>À <span class="num">w=1</span>, COMET-DA détecte les incohérences lexicales à <span class="num">.79–.87</span>,
    CometKiwi à <span class="num">.59–.66</span>. Élargir la fenêtre n'aide aucune métrique de façon soutenue,
    ré-entraînée ou non.</p>
    <a href="#q2">Voir les figures →</a>
  </article>
  <article class="ans">
    <span class="ans-q">Q3 · blocs FLORES+</span>
    <p class="ans-head">À nombre de candidats fixe, l'encodeur COMET est au hasard.</p>
    <p>Duel à 5 candidats&nbsp;: erreur <span class="num">.84 → .96</span> de k=2 à k=5, pour un hasard à
    <span class="num">.83</span>. LaBSE et E5 restent à <span class="num">.29–.59</span>. Le cœur apparié
    montre que c'est la longueur seule.</p>
    <a href="#q3">Voir les figures →</a>
  </article>
</div>
</div></header>
<main class="wrap">""")

    # ── Q1 ──────────────────────────────────────────────────
    a("""<section id="q1">
<div class="sec-head"><span class="sec-num">Q1</span>
<div><h2>Qui corrèle le mieux avec le jugement humain&nbsp;?</h2>
<p class="sec-sub">Kendall τ sur les jeux tenus hors entraînement&nbsp;: fenêtres MQM WMT22 (en-de, en-ru,
zh-en) pour k≥1, paragraphes MQM WMT23/24 et documents ESA WMT25 pour k=0.</p></div></div>

<div class="prose">
<div class="box">
<h4>Comment lire les noms de modèles</h4>
<p>Les huit bras ré-entraînés diffèrent par <b>une seule chose</b>&nbsp;: la composition de leur mélange
d'entraînement, à taille et budget constants (24&nbsp;000 lignes, 6 époques).</p>
<ul class="tight">
<li><b>« texte long »</b> — aucune phrase isolée&nbsp;: moitié fenêtres agrégées, moitié documents natifs
WMT25. Code&nbsp;: <span class="mono">frac000</span>.</li>
<li><b>« phrases »</b> — 100 % de phrases isolées (segments MQM WMT22). Code&nbsp;: <span class="mono">frac100</span>.</li>
<li><b>« gelé »</b> — encodeur figé, seule la tête de régression apprend.</li>
<li><b>DA</b> = avec référence (COMET-DA)&nbsp;; <b>Kiwi</b> = sans référence (CometKiwi)&nbsp;;
<b>« publié »</b> = la métrique distribuée, non ré-entraînée.</li>
</ul>
</div>

<p class="lede">Les dix modèles se rangent en deux blocs&nbsp;: <strong>toute la famille COMET-DA passe devant
toute la famille CometKiwi</strong>, dans les trois régimes. Le classement interne se joue à deux ou trois
millièmes — et suit la composition du mélange dans le sens attendu.</p>
</div>""")

    a(plate(1, "q1_regimes", "Les trois régimes, classés",
            "Trois panneaux de barres horizontales : Kendall tau des dix modèles sur phrases seules, "
            "fenêtres concaténées et documents natifs",
            "Entraîner sur des phrases aide sur des phrases (.410)&nbsp;; entraîner sur du texte long aide sur "
            "les fenêtres (.394) et les documents (.288). Les huit bras battent leur base partout&nbsp;: pas "
            "d'oubli à ce budget."))

    a(plate(2, "q1_tau_vs_k", "Le profil en longueur, segment par segment",
            "Deux panneaux de courbes : Kendall tau en fonction du nombre de segments par entrée",
            "L'ordre « texte long » / « phrases » s'inverse entre k=1 et k=6 chez DA. Les arcs suivent la cible, "
            "pas la métrique&nbsp;: moyenner les scores de segments débruite le jugement humain."))

    a(plate(3, "q1_docsets", "Documents natifs, jeu par jeu",
            "Carte de chaleur des dix modèles sur huit jeux de test au niveau document",
            "La moyenne cache une forte dispersion, de <span class='num'>.44</span> (wmt23 en-de) à "
            "<span class='num'>.11</span> (wmt25 en-zh). Un seul jeu, wmt25 en-uk, favorise CometKiwi."))

    a("""<div class="box">
<h4>Ce qu'on peut conclure, et ce qu'on ne peut pas</h4>
<ul class="tight">
<li><b>Solide.</b> DA &gt; Kiwi dans les trois régimes ; chaque bras bat sa base à chaque k ; la chute
phrase → document (−31 % de τ pour DA).</li>
<li><b>Fragile.</b> L'écart entre bras (≤ .005) est du même ordre que le bruit ; une graine par bras.</li>
<li><b>Interdit.</b> Comparer la colonne k=0 aux colonnes k≥1 : ni les mêmes jeux, ni les mêmes langues.</li>
</ul>
</div>
</section>""")

    # ── Q2 ──────────────────────────────────────────────────
    a("""<section id="q2">
<div class="sec-head"><span class="sec-num">Q2</span>
<div><h2>Que détectent-elles sur MetaDocEval&nbsp;?</h2>
<p class="sec-sub">Cadre contrastif de Dahan, Bawden &amp; Yvon (EAMT 2026)&nbsp;: 8&nbsp;705 paires
« original / perturbé » sur des documents WMT24++ en→{fr, es, de}, scorées par fenêtres SLIDE(w,1).</p></div></div>

<div class="prose">
<p class="lede">Sur les perturbations lexicales — le seul terrain qui discrimine — <strong>COMET-DA dépasse
CometKiwi de 15 à 28 points</strong>. Mais l'axe que le cadre est fait pour tester, la fenêtre de
contexte&nbsp;<em>w</em>, ne récompense personne&nbsp;: <strong>aucune métrique ne progresse de façon soutenue
quand on lui donne plus de contexte.</strong></p>

<p>Le dépôt ne publie aucune ligne de base — les chiffres sont réservés à l'article — et fixe seulement le
protocole&nbsp;: scorer <span class="mono">sys</span> et <span class="mono">sys_perturbed</span> avec la même
métrique, regrouper par document, concaténer les segments consécutifs pour les fenêtres 3, 6 et 9. Notre
script importe <span class="mono">scripts/load_data.py</span> du dépôt tel quel (commit
<span class="mono">d3e8dc6</span>) et re-lit le README pour recompter les paires.</p>
</div>

<div class="box flag">
<h4>Trois pièges vérifiés contre les données</h4>
<ul class="tight">
<li><b><span class="mono">sentence_splitting</span> préserve la qualité par construction</b> — son
« exactitude » est un taux de faux positifs, jamais dans les moyennes.</li>
<li><b><span class="mono">doc_id</span> est local à chaque catégorie</b>, pas global.</li>
<li><b>Le README amont sur-annonce deux catégories&nbsp;:</b> 1&nbsp;929 paires de
<span class="mono">sentence_shuffling</span> annoncées contre 1&nbsp;290 mesurées, 700 de
<span class="mono">sentence_splitting</span> contre 118. Nos chiffres utilisent les comptes mesurés.</li>
</ul>
</div>""")

    a(plate(4, "q2_baselines", "Les deux métriques publiées, catégorie par catégorie",
            "Deux panneaux de courbes : exactitude contrastive en fonction de la fenêtre de contexte",
            "Les perturbations structurelles sont saturées&nbsp;: tout se joue sur les trois catégories "
            "lexicales, où COMET-DA part haut et <b>décroît</b> avec w, et CometKiwi part bas et stagne."))

    a(plate(5, "q2_heatmap_w1", "Les dix modèles, à fenêtre d'un segment",
            "Carte de chaleur des dix modèles sur les neuf catégories de perturbation à w=1",
            "Les cinq lignes DA sont interchangeables (±.01). Les bras Kiwi gagnent <span class='num'>+.02</span> "
            "à <span class='num'>+.04</span> sur le lexical, mais paient en faux positifs sur le découpage "
            "(<span class='num'>.67</span> → <span class='num'>.74–.78</span>)."))

    a(plate(6, "q2_context", "L'entraînement sur texte long achète-t-il du discours&nbsp;?",
            "Deux panneaux : exactitude moyenne sur les trois catégories lexicales en fonction de la fenêtre",
            "Non. L'écart aux courbes grises ne se creuse pas vers la droite. <b>Les gains de corrélation de la "
            "Q1 sont au niveau du score, pas du discours.</b>"))

    a("""<div class="box flag">
<h4>Réserve sur les grandes fenêtres</h4>
<p>À w=6 et w=9 les fenêtres concaténées approchent la limite de 512 tokens de l'encodeur — davantage pour
CometKiwi, qui encode <span class="mono">mt</span> et <span class="mono">src</span> en une seule séquence.
Le script ne mesure pas encore le taux de dépassement&nbsp;: une part de la décroissance peut être de la
troncature plutôt que de la dilution.</p>
</div>
</section>""")

    # ── Q3 ──────────────────────────────────────────────────
    a("""<section id="q3">
<div class="sec-head"><span class="sec-num">Q3</span>
<div><h2>Comment les encodeurs tiennent-ils sur nos blocs FLORES+&nbsp;?</h2>
<p class="sec-sub">Des articles FLORES+ (de/es/fr/ru, pivot anglais), des blocs de k phrases consécutives,
et un négatif qui ne diffère du bloc correct que par <em>une</em> erreur injectée — causalité, entité ou
nombre, taxonomie xSIM++.</p></div></div>

<div class="prose">
<div class="box">
<h4>Le nombre de candidats est un paramètre, pas une conséquence de k</h4>
<p>Un bloc de k phrases admet jusqu'à 3k perturbations&nbsp;: si on jetait toutes les copies perturbées dans
un même pool, les blocs longs auraient mécaniquement plus de concurrents et la longueur serait confondue
avec la difficulté. Le protocole fixe donc le budget de candidats à part&nbsp;:</p>
<ul class="tight">
<li><b>Duel à 5 candidats</b> (<span class="mono">run_duel.py</span>) — le bloc correct contre
<b>exactement 5</b> négatifs à une erreur. Le score est la moyenne exacte sur <b>tous</b> les tirages de 5
négatifs parmi les m disponibles, en forme close&nbsp;: <span class="mono">C(w,5)/C(m,5)</span>, où w est le
nombre de négatifs battus. Aucune énumération, aucun échantillonnage. Un bloc avec m &lt; 5 est écarté et
compté.</li>
<li><b>Duel à 1 candidat</b> — le bloc correct contre une seule copie perturbée, tous les négatifs servant
à tour de rôle&nbsp;: couverture 100 %, aucune sélection de blocs.</li>
</ul>
<p>L'erreur xSIM++ sur pool complet, elle, dépend de k et n'est pas reportée ici.</p>
</div>

<p class="lede">À budget de candidats fixe, les deux familles ne jouent pas le même jeu.
<strong>L'encodeur de COMET est au niveau du hasard dès k=2 et au-dessus ensuite</strong> (erreur .84 → .96
pour un hasard à .83). <strong>LaBSE et E5 restent largement exploitables</strong> (.29 → .59). Le
ré-entraînement sur texte long ne répare pas ça.</p>
</div>""")

    a(plate(7, "q3_duel", "Le duel à candidats fixes",
            "Erreur du duel à 5 candidats en fonction du nombre de phrases par bloc, et part des blocs éligibles",
            "Le panneau de droite est la contrepartie du protocole&nbsp;: exiger 5 négatifs disponibles écarte "
            "86 % des blocs à k=2 contre 28 % à k=5. Les colonnes ne portent donc pas sur les mêmes blocs — "
            "le duel à 1 candidat, ci-dessous, n'a pas ce défaut."))

    a(plate(8, "q3_detection", "Le duel à 1 candidat, couverture totale",
            "Deux panneaux : taux de détection en fonction du nombre de phrases par bloc",
            "Même verdict sur l'ensemble des blocs&nbsp;: <span class='num'>.85</span> contre "
            "<span class='num'>.34</span> à k=5. Les trois variantes de COMET restent groupées&nbsp;; le "
            "ré-entraînement WMT gagne <span class='num'>+.02</span> à <span class='num'>+.03</span>, positif "
            "dans les 16 cellules (langue × k) mais loin de restaurer quoi que ce soit."))

    a(plate(9, "q3_matched_core", "Le cœur apparié&nbsp;: la longueur, toutes choses égales par ailleurs",
            "Quatre panneaux : taux de détection en fonction de la longueur totale, pour quatre remplissages",
            "Le panneau <b>inerte</b> est décisif&nbsp;: une phrase neutre répétée n'ajoute aucune sémantique, et "
            "COMET tombe pourtant de <span class='num'>.73</span> à <span class='num'>.47</span> entre L=60 et "
            "L=120. LaBSE ne descend jamais sous <span class='num'>.88</span>."))

    a(plate(10, "q3_lsi", "Le coût de la longueur, en un chiffre",
            "Diagramme en barres : indice de sensibilité à la longueur par encodeur et par remplissage",
            "COMET perd <span class='num'>8</span> à <span class='num'>12</span> points par doublement de L, "
            "LaBSE <span class='num'>1</span> à <span class='num'>2</span>. Le contraste inerte / naturel sépare "
            "la longueur seule (<span class='num'>−.105</span>) de la dérive sémantique "
            "(<span class='num'>−.123</span>)."))

    a("""<div class="tablewrap"><table>
<thead><tr><th>Encodeur</th><th>Duel 5 cand. k=2</th><th>k=5</th><th>Duel 1 cand. k=2</th><th>k=3</th><th>k=5</th><th>LSI inerte</th></tr></thead>
<tbody>
<tr><td class="rowlab">E5 multilingue</td><td class="n win">.29</td><td class="n win">.54</td><td class="n win">.94</td><td class="n win">.91</td><td class="n win">.85</td><td class="n">−.012</td></tr>
<tr><td class="rowlab">LaBSE</td><td class="n">.31</td><td class="n">.59</td><td class="n">.93</td><td class="n">.89</td><td class="n">.83</td><td class="n">−.015</td></tr>
<tr><td class="rowlab">Encodeur COMET</td><td class="n">.84</td><td class="n">.96</td><td class="n">.59</td><td class="n">.47</td><td class="n">.34</td><td class="n">−.105</td></tr>
<tr><td class="rowlab">&nbsp;&nbsp;+ finetuning Bio-MQM</td><td class="n">.85</td><td class="n">.96</td><td class="n">.58</td><td class="n">.46</td><td class="n">.34</td><td class="n">−.112</td></tr>
<tr><td class="rowlab">&nbsp;&nbsp;+ ré-entraîné texte long</td><td class="n">—</td><td class="n">—</td><td class="n">.59</td><td class="n">.48</td><td class="n">.36</td><td class="n">—</td></tr>
<tr><td class="rowlab">&nbsp;&nbsp;+ ré-entraîné phrases</td><td class="n">—</td><td class="n">—</td><td class="n">.61</td><td class="n">.50</td><td class="n">.37</td><td class="n">—</td></tr>
<tr><td class="rowlab">XLM-R (mean-pool)</td><td class="n">.90</td><td class="n">.96</td><td class="n">.51</td><td class="n">.48</td><td class="n">.44</td><td class="n">−.029</td></tr>
</tbody></table></div>
<p class="plate-cap">Duel à 5 candidats&nbsp;: erreur, hasard <span class="num">.83</span>, plus bas est mieux.
Duel à 1 candidat&nbsp;: détection, hasard <span class="num">.50</span>, plus haut est mieux. Les bras
ré-entraînés n'ont pas encore été passés au duel à 5 candidats ni au cœur apparié.</p>
</section>""")

    # ── caveats + provenance ────────────────────────────────
    a("""<section id="reserves">
<div class="sec-head"><span class="sec-num">§</span>
<div><h2>Ce qui limite la lecture</h2></div></div>
<div class="prose">
<ul class="tight">
<li><b>Les bras Kiwi s'entraînaient sur des entrées tronquées.</b> CometKiwi encode
<span class="mono">mt</span> et <span class="mono">src</span> en une seule séquence coupée à 512 tokens,
alors que le filtre plafonnait chaque côté à 480&nbsp;: 7,8 % des lignes natives et 10,7 % des lignes k=6
perdaient la fin de leur source. Les bras DA ne sont pas touchés. Les chiffres Kiwi sont des <b>bornes
inférieures</b>&nbsp;; le correctif est déployé, la vague se relance sur les données v2.</li>
<li><b>Huit bras sur trente-six.</b> Seules les deux extrémités du balayage (0 % et 100 % de phrases) ont
tourné sous le protocole corrigé. La courbe de restauration entre les deux n'est pas mesurée.</li>
<li><b>Les balayages du 14 et du 17 août sont invalides et exclus.</b> COMET ne lit que
<span class="mono">train_data[epoch % len]</span>&nbsp;: chaque bras s'entraînait sur un seul fichier
(205 à 6&nbsp;940 lignes au lieu de 24&nbsp;000) et le classement reproduisait celui des budgets.</li>
<li><b>La composition déplace aussi les langues.</b> Les pools phrase/agrégé couvrent 3 paires, le pool
natif 13, avec une seule paire commune.</li>
<li><b>Le duel à 5 candidats sélectionne les blocs.</b> Il exige 5 négatifs disponibles, d'où 14 % de blocs
éligibles à k=2 contre 72 % à k=5&nbsp;; le duel à 1 candidat (couverture 100 %) sert de contrôle.</li>
</ul>
</div>
</section>
</main>

<footer><div class="wrap">
<div>
<h4>Données sources</h4>
<ul>
<li>Q1 — <span class="mono">results/length_training/correlation_heldout.json</span></li>
<li>Q2 — <span class="mono">results/length_training/metadoceval.json</span></li>
<li>Q3 — <span class="mono">results/block_duel/block_duel.json</span> (duel à 5 candidats),
<span class="mono">results/block_xsim/block_xsim_*.json</span> (duel à 1 candidat),
<span class="mono">results/matched_core/matched_core.json</span></li>
</ul>
</div>
<div>
<h4>Reproduire</h4>
<p><span class="mono">python report/figures/make_answer_figures.py</span> → PDF + PNG dans
<span class="mono">report/figures/answers/</span>.</p>
<p><span class="mono">python report/make_answer_page.py</span> régénère cette page.</p>
<p>Protocoles&nbsp;: <span class="mono">experiments/length_isolation/</span>,
<span class="mono">experiments/length_training/</span>.</p>
</div>
<div>
<h4>Références</h4>
<p>Dahan, Bawden &amp; Yvon — <i>MetaDocEval</i>, EAMT 2026 (HAL hal-05663019).</p>
<p>Chen et al. — <i>xSIM++</i>, ACL 2023. Raunak et al. — <i>SLIDE</i>, 2024.</p>
<p>Rei et al. — <i>COMET</i> / <i>CometKiwi</i>, WMT 2022.</p>
</div>
</div></footer>""")

    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
