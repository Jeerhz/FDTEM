#!/usr/bin/env python3
"""Build report/deck_length.html — slide deck for the length-composition wave.

Figures come from report/figures/{length,align}/ and are embedded as data URIs,
so the file is self-contained and can be opened or shared as-is.

Style (CSS + keyboard navigation) is imported from make_deck_page.py rather than
copied, so the two decks cannot drift apart.

  python report/make_deck_length.py
"""
from base64 import b64encode
from pathlib import Path

from make_deck_page import HEAD, SCRIPT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/deck_length.html"
DIRS = {"length": ROOT / "report/figures/length",
        "align": ROOT / "report/figures/align"}


def img(where, name, alt):
    path = DIRS[where] / f"{name}.png"
    if not path.exists():
        raise SystemExit(f"missing figure {path} — run report/figures/"
                         f"make_{'align' if where == 'align' else 'length'}_figures.py")
    data = b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def slide_fig(num, where, name, title, alt, takeaway):
    return f"""<section class="slide">
  <header><span class="eyebrow">Figure {num}</span><h2>{title}</h2></header>
  <div class="plate">{img(where, name, alt)}</div>
  <p class="takeaway">{takeaway}</p>
</section>"""


def build():
    # The deck answers one question, so that question is its name — it also has
    # to be identifiable next to the other decks in this repo, which a label like
    # "Composition et longueur" would not be.
    p = [HEAD.replace("<title>Longueur et métriques MT</title>",
                      "<title>Sur quoi entraîner une métrique</title>"),
         '<div id="deck">']
    a = p.append

    # ── 1. title ─────────────────────────────────────────────────────────────
    a("""<section class="slide title">
  <div class="rule"></div>
  <h1>Sur quoi faut-il entraîner une métrique&nbsp;?</h1>
  <p class="sub">COMET est entraîné sur des phrases isolées et utilisé sur des documents.
  Nous ré-entraînons la même métrique sur quatre types de texte, à composition
  seule variable, et nous mesurons ce que chacun coûte et rapporte.</p>
  <div class="meta"><span>FDTEM</span><span>1<sup>er</sup> septembre 2026</span>
  <span>8 bras · 24 000 lignes chacun</span></div>
</section>""")

    # ── 2. the question ──────────────────────────────────────────────────────
    a("""<section class="slide">
  <header><span class="eyebrow">Le problème</span><h2>Un décalage entre l'entraînement et l'usage</h2></header>
  <ol class="steps">
    <li><span class="n">01</span><span><b>COMET note des phrases.</b> Ses données d'entraînement
    sont des segments isolés jugés par des humains — une phrase, une note.</span></li>
    <li><span class="n">02</span><span><b>On l'utilise sur des paragraphes et des documents.</b>
    Rien dans la métrique ne dit ce qui se passe quand l'entrée s'allonge.</span></li>
    <li><span class="n">03</span><span><b>Deux choses peuvent se dégrader, et il faut les
    séparer.</b> L'accord avec le jugement humain sur texte long, et la capacité à
    repérer une erreur diluée dans du contexte.</span></li>
  </ol>
  <p class="aside">La variable expérimentale est la <b>composition</b> du mélange
  d'entraînement, rien d'autre : même nombre de lignes, même budget, même schéma
  d'optimisation. Seul change ce dont le texte est fait.</p>
</section>""")

    # ── 3. the four arms ─────────────────────────────────────────────────────
    a("""<section class="slide">
  <header><span class="eyebrow">Le dispositif</span><h2>Quatre bras, une seule différence</h2></header>
  <div class="tablewrap"><table>
    <caption>Chaque bras part de la métrique publiée et continue son entraînement sur
    24 000 lignes — même taille, même budget (60 époques max, patience 10).</caption>
    <thead><tr><th>Bras</th><th>Texte d'entraînement</th><th>La question posée</th></tr></thead>
    <tbody>
      <tr><td><b>phrases</b></td><td class="n">100 % segments isolés</td>
          <td>le témoin : continuer sans rien changer à la longueur</td></tr>
      <tr><td><b>phrases concaténées</b></td><td class="n">100 % fenêtres d'un même document</td>
          <td>la longueur <i>synthétique</i> suffit-elle&nbsp;?</td></tr>
      <tr><td><b>documents natifs</b></td><td class="n">100 % documents entiers annotés</td>
          <td>le vrai texte long fait-il mieux, et différemment&nbsp;?</td></tr>
      <tr><td><b>mixte</b></td><td class="n">50 / 50 concaténées + natifs</td>
          <td>le mélange bat-il chacune des deux sources&nbsp;?</td></tr>
    </tbody>
  </table></div>
  <p class="aside">Le tout en double : <b>COMET-DA</b> (avec référence) et <b>CometKiwi</b>
  (sans référence), sur des lignes identiques octet pour octet — la seule différence
  entre les deux familles est la métrique de départ.</p>
</section>""")

    # ── 4. headline result ───────────────────────────────────────────────────
    a(slide_fig(1, "length", "tau_by_regime_heldout",
                "Accord avec le jugement humain, par régime d'entrée",
                "Trois panneaux — phrases seules, phrases concaténées, documents entiers — "
                "avec une barre de Kendall tau par modèle",
                "Le compromis apparaît enfin. <b>Sur documents entiers, le bras mixte passe de "
                ".272 (publié) à .318</b> ; sur phrases, c'est le bras entraîné sur phrases qui "
                "gagne, et le mixte qui perd le plus. Entraîner long achète du document et coûte "
                "de la phrase — la question devient un arbitrage, pas un progrès gratuit."))

    # ── 5. what the correlation hides ────────────────────────────────────────
    a(slide_fig(2, "length", "spread_by_k_heldout",
                "Ce que la corrélation ne peut pas voir",
                "Écart interquartile des scores rapporté à celui des phrases isolées, "
                "en fonction de la taille de la fenêtre",
                "Le tau de Kendall est invariant à toute transformation monotone : une métrique "
                "peut garder son rang et perdre sa <b>résolution</b>. C'est ce qui arrive. "
                "<b>La métrique publiée resserre ses scores sur les textes longs (×0,60), et le "
                "bras « documents natifs » encore plus (×0,47)</b> — alors que les bras phrases "
                "et concaténées élargissent leur échelle. Deux traductions doivent donc différer "
                "davantage pour que l'écart de score reste lisible."))

    # ── 6. token axis ────────────────────────────────────────────────────────
    a(slide_fig(3, "length", "tokens_heldout",
                "La longueur en tokens, pas en phrases",
                "Kendall tau et écart interquartile par tranche de tokens XLM-R, "
                "pour les deux familles",
                "Une fenêtre de 6 phrases courtes et un document natif diffèrent d'un ordre de "
                "grandeur en tokens : compter les phrases confond « plus de phrases » et « plus "
                "de texte ». Sur l'axe que le modèle voit réellement, <b>les bras se séparent "
                "bien plus nettement en résolution (bas) qu'en corrélation (haut)</b>. "
                "Les tranches de moins de 100 lignes ne sont pas tracées."))

    # ── 7. the alignment experiment — setup ──────────────────────────────────
    a("""<section class="slide">
  <header><span class="eyebrow">Deuxième expérience</span><h2>Aligner avec la métrique, pas avec l'encodeur</h2></header>
  <ol class="steps">
    <li><span class="n">01</span><span>On prend des articles FLORES+, on concatène
    <span class="mono">k</span> phrases en un bloc, et on perturbe <b>une seule phrase</b>
    du bloc : un lien causal, une entité ou un nombre.</span></li>
    <li><span class="n">02</span><span>On demande au modèle de retrouver la bonne traduction
    parmi le bloc correct et ses variantes perturbées. <b>Aucun autre article
    dans les candidats</b> : les distinguer est une compétence différente et facile,
    qui gonflerait le résultat.</span></li>
    <li><span class="n">03</span><span>Deux règles de décision sur le <b>même</b> checkpoint :
    la similarité cosinus de l'encodeur, et le score COMET lui-même. La différence isole
    ce que la tête de régression ajoute à la représentation.</span></li>
  </ol>
  <p class="aside">Le nombre de candidats est fixé par construction et ne varie pas avec
  <span class="mono">k</span> — sinon la difficulté de la tâche augmenterait avec la longueur,
  et on ne mesurerait plus la dilution.</p>
</section>""")

    a(slide_fig(4, "align", "align_protocols",
                "Le score résiste, l'encodeur se dilue",
                "Exactitude du duel et de la liste courte en fonction du nombre de phrases par bloc",
                "<b>Le score COMET est à .99 et parfaitement plat de k=1 à k=5</b> : la tête est "
                "saturée, elle repère l'erreur aussi bien dans cinq phrases que dans une. "
                "<b>L'encodeur, lui, se dilue</b> — et c'est là que le ré-entraînement agit : "
                "COMET-DA publié passe de .81 à .40, le bras « concaténées » de .86 à .55. "
                "L'entraînement sur texte long déplace la représentation, pas la tête."))

    a(slide_fig(5, "align", "align_by_category",
                "Par type de perturbation",
                "Exactitude du duel par catégorie — causalité, entité, nombre",
                "Le verdict tient dans les trois catégories. Le cosinus de CometKiwi est "
                "<b>sous le hasard</b> et y descend : ce n'est pas un bug — son espace reste "
                "aligné, mais tout y est au-dessus de .95, donc la marge utile est de ~.02 et "
                "une édition d'un mot n'y survit pas. CometKiwi n'a pas d'encodeur de phrase "
                "isolée, sa courbe cosinus est un usage dégénéré du modèle et est tracée comme tel."))

    # ── 8. what is still running ─────────────────────────────────────────────
    a("""<section class="slide">
  <header><span class="eyebrow">En cours</span><h2>Ce qui manque encore</h2></header>
  <ol class="steps">
    <li><span class="n">01</span><span><b>MetaDocEval par phénomène.</b> La moyenne sur les
    catégories laisse un phénomène qui s'améliore compenser un phénomène qui se dégrade,
    et sort plate. Un panneau par perturbation le sépare. Le calcul tourne&nbsp;: les dix
    modèles sur un GPU lent, ~1 h chacun.</span></li>
    <li><span class="n">02</span><span><b>Le bras non contrôlé.</b> Entraîné sur toutes les
    données disponibles, jeux d'évaluation compris. Sa corrélation mesure de la
    mémorisation et n'est <i>pas</i> un résultat — il borne ce que l'entraînement continu
    peut déplacer, ce qui rend lisible l'ampleur des effets des bras contrôlés.
    MetaDocEval reste propre pour lui.</span></li>
    <li><span class="n">03</span><span><b>Un doute à lever.</b> Les deux bras « documents
    natifs » s'arrêtent à l'époque 8, contre 22-28 pour les autres. Ils saturent tôt — mais
    avec une patience de 10 sur un pic précoce, on ne peut pas exclure un arrêt prématuré.
    Un relancement de ce seul bras trancherait.</span></li>
  </ol>
</section>""")

    a("</div>\n<div id=\"counter\"></div>\n<div id=\"hint\">← → pour naviguer</div>")
    a(SCRIPT)

    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
