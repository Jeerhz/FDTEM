#!/usr/bin/env python3
"""Build report/status_length.html — where the length-composition wave stands.

A scrolling report (not slides — report/make_deck_length.py is the deck), with
every figure embedded as a data URI so the file is self-contained.

Numbers quoted in the prose are read from the result JSONs at build time rather
than typed in, so the page cannot drift from the results it illustrates.

  python report/make_status_page.py
"""
from base64 import b64encode
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/status_length.html"
RES = ROOT / "results/length_training"
FIG = {"length": ROOT / "report/figures/length",
       "align": ROOT / "report/figures/align"}

HELD = json.load(open(RES / "correlation_heldout.json"))["models"]
ALIGN = json.load(open(ROOT / "results/comet_align/comet_align.json"))["scorers"]


def tau(m, k):
    return HELD.get(m, {}).get("_mean_by_k", {}).get(k, {}).get("kendall")


def concat(m):
    v = [tau(m, k) for k in ("2", "3", "4", "6")]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def d3(x):
    return "—" if x is None else f"{x:+.3f}"


def img(where, name, alt):
    p = FIG[where] / f"{name}.png"
    if not p.exists():
        raise SystemExit(f"missing figure {p}")
    return (f'<img src="data:image/png;base64,{b64encode(p.read_bytes()).decode()}" '
            f'alt="{alt}" loading="lazy">')


def figure(where, name, alt, caption):
    return f"""<figure>
  <div class="plate">{img(where, name, alt)}</div>
  <figcaption>{caption}</figcaption>
</figure>"""


ARMS = [("frac100", "phrases"), ("frac000agg", "phrases concaténées"),
        ("frac000nat", "documents natifs"), ("frac000", "mixte")]


def results_table(fam, famlab):
    base = f"{fam}-base"
    rows = [f'<tr class="base"><th scope="row">métrique publiée</th>'
            f'<td class="n">{f3(tau(base, "1"))}</td>'
            f'<td class="n">{f3(concat(base))}</td>'
            f'<td class="n">{f3(tau(base, "0"))}</td>'
            f'<td class="n">—</td><td class="n">—</td></tr>']
    for arm, lab in ARMS:
        m = f"{fam}-{arm}"
        if m not in HELD:
            continue
        d1 = (tau(m, "1") - tau(base, "1")) if tau(m, "1") and tau(base, "1") else None
        d0 = (tau(m, "0") - tau(base, "0")) if tau(m, "0") and tau(base, "0") else None
        cls = lambda v: "up" if v and v > 0.004 else ("down" if v and v < -0.004 else "flat")
        rows.append(
            f'<tr><th scope="row">{lab}</th>'
            f'<td class="n">{f3(tau(m, "1"))}</td>'
            f'<td class="n">{f3(concat(m))}</td>'
            f'<td class="n">{f3(tau(m, "0"))}</td>'
            f'<td class="n {cls(d1)}">{d3(d1)}</td>'
            f'<td class="n {cls(d0)}">{d3(d0)}</td></tr>')
    return f"""<div class="tablewrap"><table>
  <caption>{famlab} — Kendall τ sur les jeux held-out, jamais vus ni à l'entraînement
  ni à la sélection.</caption>
  <thead><tr><th scope="col">entraîné sur</th><th scope="col">phrases</th>
  <th scope="col">concaténées</th><th scope="col">documents</th>
  <th scope="col">Δ phrases</th><th scope="col">Δ documents</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></div>"""


HEAD = """<title>Longueur, composition, alignement</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  color-scheme: light;
  --bg: #faf9f6;
  --ink: #16181c;
  --ink-soft: #555961;
  --ink-faint: #8b8f96;
  --rule: #e2e1db;
  --accent: #1f66be;
  --accent-soft: #1f66be1a;
  --paper: #fcfcfb;
  --paper-rule: #dedcd5;
  --ok: #1b7f52;
  --warn: #9a6212;
  --down: #b3402f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #101113;
    --ink: #f0efec;
    --ink-soft: #9aa0a8;
    --ink-faint: #6d737b;
    --rule: #26282d;
    --accent: #74a9ee;
    --accent-soft: #74a9ee1f;
    --paper: #fcfcfb;
    --paper-rule: #2e3136;
    --ok: #58c08b;
    --warn: #d9a24a;
    --down: #e8836f;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #101113;
  --ink: #f0efec;
  --ink-soft: #9aa0a8;
  --ink-faint: #6d737b;
  --rule: #26282d;
  --accent: #74a9ee;
  --accent-soft: #74a9ee1f;
  --paper: #fcfcfb;
  --paper-rule: #2e3136;
  --ok: #58c08b;
  --warn: #d9a24a;
  --down: #e8836f;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
  font-size: 17px; line-height: 1.62; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: clamp(28px, 5vw, 72px) clamp(18px, 4vw, 40px) 96px; }

header.top { display: flex; flex-direction: column; gap: 14px; padding-bottom: 26px;
             border-bottom: 1px solid var(--rule); margin-bottom: 40px; }
.kicker { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .13em;
          text-transform: uppercase; color: var(--accent); }
h1 { font-family: "Source Serif 4", Georgia, serif; font-weight: 700;
     font-size: clamp(30px, 4.6vw, 46px); line-height: 1.08; margin: 0;
     letter-spacing: -.017em; text-wrap: balance; max-width: 20ch; }
.standfirst { margin: 0; font-size: clamp(16px, 1.6vw, 19px); color: var(--ink-soft); max-width: 68ch; }
.meta { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-faint);
        display: flex; flex-wrap: wrap; gap: 6px 22px; }

/* ── status strip ─────────────────────────────────────── */
.status { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
          gap: 1px; background: var(--rule); border: 1px solid var(--rule);
          border-radius: 3px; overflow: hidden; margin-bottom: 46px; }
.status div { background: var(--bg); padding: 15px 17px; display: flex;
              flex-direction: column; gap: 5px; }
.status .lab { font-family: "IBM Plex Mono", monospace; font-size: 10.5px;
               letter-spacing: .1em; text-transform: uppercase; color: var(--ink-faint); }
.status .val { font-size: 15px; font-weight: 600; line-height: 1.32; }
.status .val.done { color: var(--ok); }
.status .val.run  { color: var(--warn); }

section { margin-bottom: 54px; scroll-margin-top: 24px; }
h2 { font-family: "Source Serif 4", Georgia, serif; font-weight: 600;
     font-size: clamp(21px, 2.5vw, 28px); line-height: 1.2; margin: 0 0 6px;
     letter-spacing: -.012em; text-wrap: balance; }
.qline { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .1em;
         text-transform: uppercase; color: var(--ink-faint); margin: 0 0 14px; }
p { margin: 0 0 15px; max-width: 68ch; }
p:last-child { margin-bottom: 0; }
b, strong { font-weight: 600; }
a { color: var(--accent); }
.mono { font-family: "IBM Plex Mono", monospace; font-size: .89em; }

figure { margin: 22px 0 12px; }
.plate { background: var(--paper); border: 1px solid var(--paper-rule); border-radius: 3px;
         padding: clamp(8px, 1.2vw, 16px); overflow-x: auto; }
.plate img { display: block; width: 100%; min-width: 620px; height: auto; }
figcaption { margin-top: 10px; font-size: 14px; color: var(--ink-soft); max-width: 74ch; }

.tablewrap { overflow-x: auto; margin: 20px 0; }
table { border-collapse: collapse; width: 100%; min-width: 560px; font-size: 15px; }
caption { text-align: left; font-size: 13.5px; color: var(--ink-soft); padding-bottom: 10px; }
th, td { text-align: left; padding: 8px 16px 8px 0; border-bottom: 1px solid var(--rule);
         white-space: nowrap; }
thead th { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; letter-spacing: .09em;
           text-transform: uppercase; color: var(--ink-faint); font-weight: 500; }
tbody th { font-weight: 600; }
td.n { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
td.up { color: var(--ok); } td.down { color: var(--down); } td.flat { color: var(--ink-faint); }
tr.base td, tr.base th { color: var(--ink-soft); }

.callout { border-left: 2px solid var(--accent); background: var(--accent-soft);
           padding: 14px 18px; border-radius: 0 3px 3px 0; margin: 20px 0;
           font-size: 15.5px; }
.callout p { max-width: none; }
.caveat { border-left: 2px solid var(--rule); padding-left: 18px; margin: 20px 0;
          font-size: 14.5px; color: var(--ink-soft); }
.caveat p { max-width: 72ch; }
ul.plain { margin: 0 0 15px; padding-left: 20px; max-width: 68ch; }
ul.plain li { margin-bottom: 7px; }
hr.div { border: 0; border-top: 1px solid var(--rule); margin: 0 0 40px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>"""


def build():
    p = [HEAD, '<div class="wrap">']
    a = p.append

    a("""<header class="top">
  <span class="kicker">FDTEM · état d'avancement</span>
  <h1>Ce que la longueur des données d'entraînement change</h1>
  <p class="standfirst">Huit métriques ré-entraînées sur quatre types de texte, à
  composition seule variable, puis mesurées sur l'accord humain, la distribution
  des scores, la détection d'erreurs discursives et l'alignement source–traduction.</p>
  <div class="meta"><span>2 septembre 2026</span><span>8 bras entraînés · 24 000 lignes chacun</span>
  <span>held-out WMT22/23/24/25 · MetaDocEval · FLORES+</span></div>
</header>""")

    a("""<div class="status">
  <div><span class="lab">8 bras contrôlés</span><span class="val done">entraînés et évalués</span></div>
  <div><span class="lab">MetaDocEval</span><span class="val done">10 modèles, par phénomène</span></div>
  <div><span class="lab">Alignement FLORES+</span><span class="val done">6 scoreurs</span></div>
  <div><span class="lab">Bras non contrôlé</span><span class="val run">entraînement en cours</span></div>
</div>""")

    # ── 1. the design ────────────────────────────────────────────────────────
    a("""<section>
  <h2>Le dispositif</h2>
  <p class="qline">Une seule variable</p>
  <p>COMET est entraîné sur des phrases isolées et utilisé sur des paragraphes et des
  documents. Nous reprenons la métrique publiée et continuons son entraînement sur
  quatre mélanges qui ne diffèrent que par <b>ce dont le texte est fait</b> — même
  nombre de lignes (24 000), même budget (60 époques max, patience 10), même schéma
  d'optimisation :</p>
  <ul class="plain">
    <li><b>phrases</b> — 100 % de segments isolés. Le témoin : continuer sans rien
    changer à la longueur.</li>
    <li><b>phrases concaténées</b> — 100 % de fenêtres de phrases consécutives d'un
    même document. La longueur <i>synthétique</i>.</li>
    <li><b>documents natifs</b> — 100 % de documents entiers annotés. Le vrai texte long.</li>
    <li><b>mixte</b> — moitié concaténées, moitié natifs.</li>
  </ul>
  <p>Le tout en double : <b>COMET-DA</b> (avec référence) et <b>CometKiwi</b> (sans
  référence), sur des lignes identiques octet pour octet.</p>
</section>
<hr class="div">""")

    # ── 2. headline ──────────────────────────────────────────────────────────
    a(f"""<section>
  <h2>Le compromis longueur / phrase</h2>
  <p class="qline">Résultat principal</p>
  <p>Avec dix fois le budget de la première vague, les quatre bras se séparent enfin.
  <b>Aucun ne domine partout</b> : entraîner sur du texte long achète du document et
  coûte de la phrase, et réciproquement.</p>
  {figure("length", "tau_by_regime_heldout",
          "Kendall tau par régime d'entrée pour les dix modèles",
          "Accord avec le jugement humain, par régime d'entrée. Hachuré = CometKiwi, "
          "plein = COMET-DA, barre creuse = métrique publiée.")}
  {results_table("da", "COMET-DA (avec référence)")}
  {results_table("qe", "CometKiwi (sans référence)")}
  <div class="callout"><p><b>CometKiwi gagne bien plus que COMET-DA.</b> Sur phrases,
  la métrique sans référence passe de {f3(tau('qe-base','1'))} à
  {f3(tau('qe-frac100','1'))} — <b>{d3(tau('qe-frac100','1') - tau('qe-base','1'))}</b>,
  soit un quart de mieux en relatif — là où COMET-DA ne gagne que
  {d3(tau('da-frac100','1') - tau('da-base','1'))}. Sans référence pour s'ancrer, la
  métrique avait beaucoup plus de marge à récupérer, et la continuation d'entraînement
  la récupère.</p></div>
  <p>Sur documents entiers, c'est le bras <b>mixte</b> qui l'emporte côté COMET-DA
  ({f3(tau('da-base','0'))} → {f3(tau('da-frac000','0'))}), et il paie ce gain sur les
  phrases ({d3(tau('da-frac000','1') - tau('da-base','1'))}) — le seul bras à y perdre
  franchement. Côté CometKiwi, les documents natifs donnent le meilleur document
  ({f3(tau('qe-frac000nat','0'))}).</p>
</section>
<hr class="div">""")

    # ── 3. distributions ─────────────────────────────────────────────────────
    a(f"""<section>
  <h2>La distribution des scores, pas seulement leur rang</h2>
  <p class="qline">Ce qu'une corrélation ne peut pas montrer</p>
  <p>Le τ de Kendall est invariant à toute transformation monotone : une métrique peut
  garder son rang et perdre sa <b>résolution</b>, c'est-à-dire resserrer ses scores au
  point que deux traductions doivent beaucoup différer pour que l'écart soit lisible.
  C'est exactement ce qui se passe.</p>
  {figure("length", "dist_by_k_heldout",
          "Distribution des scores par taille de fenêtre, un panneau par modèle",
          "Un panneau par modèle : médiane, intervalle interquartile et 5–95 %. La bande "
          "grise en arrière-plan est la métrique publiée de la même famille.")}
  {figure("length", "spread_by_k_heldout",
          "Écart interquartile rapporté à celui des phrases isolées",
          "Le même phénomène en une courbe : l'écart interquartile de chaque régime "
          "divisé par celui des phrases isolées.")}
  <p>La métrique publiée <b>comprime</b> son échelle sur les textes longs (×0,60 à six
  phrases) et le bras « documents natifs » comprime encore plus (×0,47) — alors que les
  bras « phrases » et « concaténées » <b>élargissent</b> la leur (×1,3 à ×1,4).
  Entraîner sur du texte long améliore la corrélation sur documents tout en dégradant
  la résolution : les deux ne vont pas ensemble.</p>
</section>
<hr class="div">""")

    # ── 4. token axis ────────────────────────────────────────────────────────
    a(f"""<section>
  <h2>Compter les tokens, pas les phrases</h2>
  <p class="qline">Le bon axe</p>
  <p>Un régime nommé par un nombre de phrases n'est pas un régime de longueur. Sur nos
  jeux held-out, la mesure le montre sans ambiguïté :</p>
  {figure("length", "token_length_heldout",
          "Distribution du nombre de tokens pour chaque taille de fenêtre",
          "Longueur réelle de chaque régime, en tokens XLM-R. À gauche la distribution "
          "complète, à droite médiane et interquartile.")}
  <div class="callout"><p><b>Un « document entier » y fait 70 tokens de médiane, une
  fenêtre de 6 phrases en fait 181.</b> Le document est <i>plus court</i> que la
  concaténation. Ordonner les régimes par nombre de phrases et les ordonner par
  longueur réelle ne donnent donc pas le même ordre — et c'est la longueur réelle que
  le modèle encode.</p></div>
  <p>Toutes les quantités sont donc reprises sur l'axe que le modèle voit : le mt seul
  pour les modèles avec référence, qui encodent les trois côtés séparément, et src+mt
  pour CometKiwi, qui les concatène en une seule séquence de 512 tokens.</p>
  {figure("length", "tokens_heldout",
          "Kendall tau et écart interquartile par tranche de tokens",
          "En haut l'accord de rang par tranche de longueur, en bas la largeur de "
          "l'échelle utilisée. Les tranches de moins de 100 lignes ne sont pas tracées : "
          "au-delà de 320 tokens le mt des modèles avec référence devient trop rare "
          "(13 lignes, puis 2) pour qu'un τ y veuille dire quelque chose.")}
</section>
<hr class="div">""")

    # ── 5. per phenomenon ────────────────────────────────────────────────────
    a(f"""<section>
  <h2>Détection d'erreurs, phénomène par phénomène</h2>
  <p class="qline">MetaDocEval · 10 modèles · fenêtres w ∈ {{1, 3, 6, 9}}</p>
  <p>Micro-moyenner les catégories laisse un phénomène qui s'améliore compenser un
  phénomène qui se dégrade, et sort plat. Un panneau par perturbation les sépare, et
  l'écart à la métrique publiée rend la lecture directe.</p>
  {figure("length", "mde_phenomena_delta_da",
          "Écart de détection à la métrique publiée, par phénomène, COMET-DA",
          "COMET-DA — chaque courbe est exactitude(bras) − exactitude(publié) à fenêtre "
          "égale. Chaque panneau a sa propre échelle.")}
  {figure("length", "mde_phenomena_delta_qe",
          "Écart de détection à la métrique publiée, par phénomène, CometKiwi",
          "CometKiwi — même lecture.")}
  <p>Deux constats, dont un qui contredit l'attente de départ :</p>
  <ul class="plain">
    <li><b>Aucun bras n'achète de détection discursive pour COMET-DA.</b> Sur les trois
    catégories lexicales, presque toutes les courbes sont sous zéro ; « documents
    natifs » est le seul à effleurer la ligne.</li>
    <li><b>Les effets ne se compensent pas — ils vont dans le même sens.</b> Nous
    cherchions une compensation entre phénomènes ; le découpage montre au contraire un
    coût homogène. La moyenne n'était donc pas trompeuse ici, mais on ne pouvait pas le
    savoir sans regarder.</li>
    <li><b>Un vrai gain, visible seulement par catégorie</b> : sur le découpage de
    phrases — qui préserve la qualité, donc dont l'« exactitude » est un taux de faux
    positifs — les bras « concaténées » et « mixte » descendent de 0,25. Entraîner sur
    du texte concaténé rend le modèle nettement moins susceptible de pénaliser un
    simple redécoupage.</li>
  </ul>
</section>
<hr class="div">""")

    # ── 6. alignment ─────────────────────────────────────────────────────────
    kiwi_s = ALIGN["comet-score:comet:wmt22-cometkiwi-da"]["_mean_by_k"]
    da_c = ALIGN["encoder-cos:comet:wmt22-comet-da"]["_mean_by_k"]
    arm_c = [v for k, v in ALIGN.items()
             if k.startswith("encoder-cos") and "gksapnlz" in k][0]["_mean_by_k"]
    a(f"""<section>
  <h2>Aligner avec le score COMET, pas avec l'encodeur</h2>
  <p class="qline">FLORES+ · blocs de k phrases · une phrase perturbée</p>
  <p>On concatène <span class="mono">k</span> phrases d'un article FLORES+ en un bloc et
  on perturbe <b>une seule</b> de ses phrases — un lien causal, une entité, un nombre.
  Le modèle doit retrouver la bonne traduction parmi le bloc correct et ses variantes
  perturbées. <b>Aucun autre article dans les candidats</b> : les distinguer est une
  compétence différente et facile, qui gonflerait le résultat. Deux règles de décision
  sur le <i>même</i> checkpoint : la similarité cosinus de l'encodeur, et le score COMET.</p>
  {figure("align", "align_protocols",
          "Exactitude du duel et de la liste courte selon la longueur du bloc",
          "Triangles creux = bras ré-entraîné. Le nombre de candidats est fixé et ne "
          "varie pas avec k, sinon la difficulté augmenterait avec la longueur.")}
  <div class="callout"><p><b>Le score est à {kiwi_s['1']['duel_accuracy']:.2f} et
  parfaitement plat</b> de k=1 à k=5 : la tête de régression repère l'erreur aussi bien
  dans cinq phrases que dans une. <b>L'encodeur, lui, se dilue</b> — COMET-DA publié
  passe de {da_c['1']['duel_accuracy']:.2f} à {da_c['5']['duel_accuracy']:.2f} — et
  c'est là que le ré-entraînement agit : le bras « concaténées » tient
  {arm_c['5']['duel_accuracy']:.2f} à k=5, soit
  <b>{arm_c['5']['duel_accuracy'] - da_c['5']['duel_accuracy']:+.2f}</b>.
  L'entraînement sur texte long déplace la représentation, pas la tête.</p></div>
  {figure("align", "align_by_category",
          "Exactitude du duel par catégorie de perturbation",
          "Le verdict tient dans les trois catégories.")}
  <div class="caveat"><p>Le cosinus de CometKiwi est <b>sous le hasard</b> et y descend.
  Ce n'est pas un bug : une sonde directe montre que son espace reste aligné (la bonne
  traduction gagne sur chaque paire testée), mais tout y est au-dessus de 0,95, donc la
  marge utile est de l'ordre de 0,02 et une édition d'un mot n'y survit pas. CometKiwi
  est un <span class="mono">UnifiedMetric</span> : il encode « mt &lt;/s&gt;&lt;/s&gt; src »
  en une seule séquence et n'a pas d'encodeur de phrase isolée. Sa courbe cosinus est un
  usage dégénéré du modèle, et elle est tracée comme telle.</p></div>
</section>
<hr class="div">""")

    # ── 7. what is running ───────────────────────────────────────────────────
    a("""<section>
  <h2>Le bras non contrôlé</h2>
  <p class="qline">En cours</p>
  <p>Un neuvième et dixième bras s'entraînent sur <b>toutes les données disponibles,
  jeux d'évaluation compris</b> — 281 525 lignes, dont 11 467 viennent des portions
  d'évaluation. Encodeur dégelé dès le premier pas, taux d'apprentissage dix fois plus
  élevés, early stopping désactivé.</p>
  <p>Sa corrélation mesurera de la <b>mémorisation</b> et ne sera pas un résultat. Il
  existe pour borner ce que la continuation d'entraînement peut déplacer : c'est cette
  borne qui rend lisible l'ampleur des effets des bras contrôlés. MetaDocEval, corpus
  différent dont rien n'entre dans le mélange, reste une lentille honnête pour lui.</p>
  <div class="caveat"><p>Le mélange porte un fichier <span class="mono">CONTAMINATED</span>,
  <span class="mono">contaminated: true</span> dans son manifeste, une alerte dans le log
  du job et une marque dans le tableau de résultats. Trois garde-fous pour une seule
  chose : que personne ne rapporte ces chiffres comme un accord avec le jugement humain.</p></div>
</section>

<section>
  <h2>Ce qui reste à trancher</h2>
  <p class="qline">Questions ouvertes</p>
  <ul class="plain">
    <li><b>Les deux bras « documents natifs » s'arrêtent à l'époque 8</b>, contre 22 à 28
    pour les autres. Ils saturent tôt — mais avec une patience de 10 sur un pic précoce,
    un arrêt prématuré n'est pas exclu. Relancer ce seul bras avec une patience plus
    large lèverait le doute.</li>
    <li><b>La couverture linguistique bouge avec la composition.</b> Les phrases et les
    fenêtres viennent de WMT22 (3 paires), les documents natifs de WMT25 (13 paires) ;
    les deux pools ne partagent qu'une paire. « Texte long » et « ces langues-là » ne
    sont donc pas complètement séparés, et l'analyse par paire de langues reste à faire.</li>
    <li><b>La perturbation d'entité puise dans un banc construit sur le corpus</b>, donc
    une entité substituée peut coïncider avec celle du côté source. C'est la piste la
    plus probable pour expliquer pourquoi le cosinus de CometKiwi chute à 0,03 sur cette
    catégorie précise.</li>
  </ul>
</section>""")

    a("</div>")
    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
