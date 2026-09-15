#!/usr/bin/env python3
"""Build report/deck_evaluation.html — a minimal slide deck of the answer figures.

Figures come from report/figures/answers/ (embedded as data URIs, self-contained).

  python report/make_deck_page.py
"""
from base64 import b64encode
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "report/figures/answers"
OUT = ROOT / "report/deck_evaluation.html"


def img(name, alt):
    data = b64encode((FIGDIR / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def slide_fig(num, name, title, alt, takeaway):
    return f"""<section class="slide">
  <header><span class="eyebrow">Figure {num}</span><h2>{title}</h2></header>
  <div class="plate">{img(name, alt)}</div>
  <p class="takeaway">{takeaway}</p>
</section>"""


HEAD = """<title>Longueur et métriques MT</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  color-scheme: light;
  --bg: #faf9f6;
  --ink: #16181c;
  --ink-2: #55596180;
  --ink-soft: #555961;
  --rule: #e2e1db;
  --accent: #1f66be;
  --paper: #fcfcfb;
  --paper-rule: #dedcd5;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #101113;
    --ink: #f0efec;
    --ink-2: #9aa0a880;
    --ink-soft: #9aa0a8;
    --rule: #26282d;
    --accent: #74a9ee;
    --paper: #fcfcfb;
    --paper-rule: #2e3136;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #101113;
  --ink: #f0efec;
  --ink-2: #9aa0a880;
  --ink-soft: #9aa0a8;
  --rule: #26282d;
  --accent: #74a9ee;
  --paper: #fcfcfb;
  --paper-rule: #2e3136;
}

* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  background: var(--bg);
  color: var(--ink);
  font-family: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
#deck { height: 100dvh; overflow-y: auto; scroll-snap-type: y mandatory; scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { #deck { scroll-behavior: auto; } }

.slide {
  min-height: 100dvh; scroll-snap-align: start; scroll-snap-stop: always;
  display: flex; flex-direction: column; justify-content: center; gap: clamp(14px, 2.4vh, 30px);
  padding: clamp(28px, 5vw, 72px) clamp(24px, 6vw, 96px) clamp(46px, 7vh, 78px);
  max-width: 1500px; margin: 0 auto;
}
.slide > header { display: flex; flex-direction: column; gap: 8px; }
.eyebrow {
  font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--accent);
}
h2 {
  font-family: "Source Serif 4", Georgia, serif; font-weight: 600;
  font-size: clamp(21px, 2.9vw, 34px); line-height: 1.15; margin: 0;
  letter-spacing: -.012em; text-wrap: balance; max-width: 26ch;
}
.plate {
  background: var(--paper); border: 1px solid var(--paper-rule); border-radius: 2px;
  padding: clamp(8px, 1.4vw, 18px); overflow-x: auto;
  display: flex; align-items: center; justify-content: center;
}
.plate img { display: block; width: 100%; min-width: 560px; height: auto; max-height: 64dvh; object-fit: contain; }
.takeaway { margin: 0; font-size: clamp(14px, 1.35vw, 17px); color: var(--ink-soft); max-width: 92ch; }
.takeaway b { color: var(--ink); font-weight: 600; }

/* ── title slide ───────────────────────────────────────── */
.title { justify-content: center; gap: clamp(18px, 3vh, 34px); }
.title h1 {
  font-family: "Source Serif 4", Georgia, serif; font-weight: 600;
  font-size: clamp(36px, 6.4vw, 74px); line-height: 1.04; margin: 0;
  letter-spacing: -.02em; max-width: 16ch;
}
.title .sub { font-size: clamp(16px, 1.8vw, 21px); color: var(--ink-soft); max-width: 56ch; margin: 0; }
.title .rule { width: 64px; height: 2px; background: var(--accent); }
.title .meta {
  font-family: "IBM Plex Mono", monospace; font-size: 12.5px; color: var(--ink-soft);
  display: flex; flex-wrap: wrap; gap: 8px 24px;
}

/* ── text slides ───────────────────────────────────────── */
.steps { list-style: none; margin: 0; padding: 0; display: grid; gap: clamp(10px, 1.7vh, 20px); max-width: 88ch; }
.steps li { display: grid; grid-template-columns: 30px 1fr; gap: 18px; align-items: baseline;
            font-size: clamp(15px, 1.45vw, 18.5px); }
.steps .n { font-family: "IBM Plex Mono", monospace; font-size: 13px; color: var(--accent); }
.steps b { font-weight: 600; }
.aside {
  border-left: 2px solid var(--rule); padding-left: 18px; margin: 0;
  font-size: clamp(13.5px, 1.25vw, 16px); color: var(--ink-soft); max-width: 84ch;
}
.mono { font-family: "IBM Plex Mono", monospace; font-size: .88em; }

table { border-collapse: collapse; font-size: clamp(13px, 1.25vw, 16px); width: 100%; max-width: 1050px; }
caption { text-align: left; font-size: 13px; color: var(--ink-soft); padding-bottom: 10px; }
th, td { text-align: left; padding: 7px 16px 7px 0; border-bottom: 1px solid var(--rule); white-space: nowrap; }
thead th { font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .09em;
           text-transform: uppercase; color: var(--ink-soft); font-weight: 500; }
td.n { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
td.gap { color: var(--accent); font-family: "IBM Plex Mono", monospace; }
tbody tr.sep td { border-top: 2px solid var(--rule); }
.tablewrap { overflow-x: auto; }

/* ── chrome ────────────────────────────────────────────── */
#counter {
  position: fixed; right: clamp(16px, 3vw, 34px); bottom: clamp(14px, 2.4vh, 26px);
  font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-2);
  pointer-events: none; user-select: none;
}
#hint {
  position: fixed; left: clamp(16px, 3vw, 34px); bottom: clamp(14px, 2.4vh, 26px);
  font-family: "IBM Plex Mono", monospace; font-size: 11.5px; color: var(--ink-2);
  pointer-events: none; transition: opacity .4s;
}
#hint.gone { opacity: 0; }
</style>"""

SCRIPT = """<script>
(function () {
  var deck = document.getElementById('deck');
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var counter = document.getElementById('counter');
  var hint = document.getElementById('hint');
  function current() {
    var top = deck.scrollTop, best = 0, dist = Infinity;
    slides.forEach(function (s, i) {
      var d = Math.abs(s.offsetTop - top);
      if (d < dist) { dist = d; best = i; }
    });
    return best;
  }
  function paint() { counter.textContent = (current() + 1) + ' / ' + slides.length; }
  function go(i) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    deck.scrollTo({ top: slides[i].offsetTop, behavior: 'smooth' });
    if (hint) hint.classList.add('gone');
  }
  deck.addEventListener('scroll', paint, { passive: true });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === ' ') {
      e.preventDefault(); go(current() + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp' || e.key === 'PageUp') {
      e.preventDefault(); go(current() - 1);
    } else if (e.key === 'Home') { e.preventDefault(); go(0); }
    else if (e.key === 'End') { e.preventDefault(); go(slides.length - 1); }
  });
  paint();
})();
</script>"""


def build():
    p = [HEAD, '<div id="deck">']
    a = p.append

    # ── 1. title ────────────────────────────────────────────
    a("""<section class="slide title">
  <div class="rule"></div>
  <h1>Métriques MT à l'épreuve de la longueur</h1>
  <p class="sub">Corrélation avec le jugement humain, détection d'erreurs discursives,
  et robustesse des encodeurs quand l'entrée s'allonge.</p>
  <div class="meta"><span>FDTEM</span><span>25 août 2026</span><span>7 figures</span></div>
</section>""")

    # ── 2–4. Q1 ─────────────────────────────────────────────
    a(slide_fig(1, "q1_regimes", "Corrélation humaine : trois régimes d'entrée",
                "Barres horizontales : Kendall tau des dix modèles sur phrases seules, "
                "fenêtres concaténées et documents natifs",
                "Toute la famille COMET-DA passe devant toute la famille CometKiwi, dans les trois régimes. "
                "Entraîner sur des phrases aide sur les phrases (.410), entraîner sur du texte long aide sur "
                "les documents (.288). <b>L'écart entre bras (≤ .005) reste du même ordre que le bruit.</b>"))

    a(slide_fig(2, "q1_tau_vs_k", "Le profil en longueur, segment par segment",
                "Courbes : Kendall tau en fonction du nombre de segments par entrée",
                "Les huit bras ré-entraînés sont au-dessus de leur base à chaque k — aucun oubli. "
                "<b>La remontée à grand k vient de la cible, pas de la métrique</b> : moyenner les notes de "
                "segments débruite le jugement humain."))

    a(slide_fig(3, "q1_docsets", "Documents natifs, jeu par jeu",
                "Carte de chaleur des dix modèles sur huit jeux de test au niveau document",
                "La moyenne cache une dispersion de <span class='mono'>.44</span> à "
                "<span class='mono'>.11</span>. Un seul jeu, wmt25 en-uk, inverse le classement."))

    # ── 5. contrastive accuracy recipe ──────────────────────
    a("""<section class="slide">
  <header><span class="eyebrow">Méthode</span><h2>L'exactitude contrastive, en cinq gestes</h2></header>
  <ol class="steps">
    <li><span class="n">01</span><span><b>Une paire.</b> Un document traduit, et le même document où
    <em>une seule chose</em> a été altérée — un temps verbal, un connecteur, une phrase supprimée.
    Rien d'autre ne change.</span></li>
    <li><span class="n">02</span><span><b>On découpe.</b> Chaque version est fenêtrée en w phrases
    consécutives, au pas de 1 — SLIDE(w,1). À w=1 la métrique ne voit jamais le contexte ; à w=9 elle voit
    presque tout le document (11,3 phrases en moyenne).</span></li>
    <li><span class="n">03</span><span><b>On score chaque fenêtre</b> avec la métrique testée, à partir de
    la source, de l'hypothèse et de la référence — sans la référence pour les métriques QE.</span></li>
    <li><span class="n">04</span><span><b>On moyenne</b> les scores de fenêtres pour obtenir un score par
    document. La perturbation ne touche qu'une partie des fenêtres : c'est là qu'elle se dilue.</span></li>
    <li><span class="n">05</span><span><b>On compte.</b> L'exactitude est la part des paires où l'original
    est mieux noté que le perturbé. Hasard = 50 %. Test t apparié sur les écarts par document.</span></li>
  </ol>
  <p class="aside">Deux garde-fous. Les perturbations structurelles (suppression, répétition, permutation)
  sont saturées près de 100 % : elles ne discriminent rien. Et le découpage de phrases préserve la qualité
  par construction — son « exactitude » est un <b>taux de faux positifs</b>, à lire à l'envers.</p>
</section>""")

    a(slide_fig(5, "q2_heatmap_w1", "MetaDocEval : dix modèles à fenêtre d'un segment",
                "Carte de chaleur des dix modèles sur les neuf catégories de perturbation à w=1",
                "Les cinq lignes DA sont interchangeables (±.01). Les bras Kiwi gagnent sur le lexical mais "
                "paient en faux positifs sur le découpage (<span class='mono'>.67</span> → "
                "<span class='mono'>.74–.78</span>)."))

    a(slide_fig(6, "q2_context", "Le contexte n'achète pas de compétence discursive",
                "Courbes : exactitude moyenne sur les trois catégories lexicales en fonction de la fenêtre",
                "Si l'entraînement sur texte long achetait du discours, l'écart aux courbes grises se "
                "creuserait vers la droite. <b>Il ne se creuse pas.</b>"))

    # ── 8. verification against the paper ───────────────────
    a("""<section class="slide">
  <header><span class="eyebrow">Vérification</span>
  <h2>Nos chiffres MetaDocEval ne reproduisent pas ceux du papier</h2></header>
  <div class="tablewrap">
  <table>
    <caption>Exactitude contrastive, micro-moyennée sur 3 paires de langues et 2 systèmes.
    Papier : Dahan, Bawden &amp; Yvon (EAMT 2026), §6.1–6.2. Nous : commit d3e8dc6 du jeu de test publié.</caption>
    <thead><tr><th>Catégorie</th><th>Métrique</th><th>Papier w=1</th><th>Nous w=1</th><th>Papier w=9</th><th>Nous w=9</th><th>Écart</th></tr></thead>
    <tbody>
      <tr><td>cohérence lexicale</td><td>COMET</td><td class="n">.64</td><td class="n">.87</td><td class="n">.56</td><td class="n">.75</td><td class="gap">+.23</td></tr>
      <tr><td>connecteurs</td><td>COMET</td><td class="n">≈.50</td><td class="n">.82</td><td class="n">≈.50</td><td class="n">.69</td><td class="gap">+.32</td></tr>
      <tr><td>temps verbaux</td><td>COMET</td><td class="n">≈.50</td><td class="n">.79</td><td class="n">≈.50</td><td class="n">.74</td><td class="gap">+.29</td></tr>
      <tr class="sep"><td>cohérence lexicale</td><td>CometKiwi</td><td class="n">.38</td><td class="n">.59</td><td class="n">.58</td><td class="n">.73</td><td class="gap">+.21</td></tr>
      <tr><td>connecteurs</td><td>CometKiwi</td><td class="n">.34</td><td class="n">.64</td><td class="n">.50</td><td class="n">.65</td><td class="gap">+.30</td></tr>
      <tr><td>temps verbaux</td><td>CometKiwi</td><td class="n">.32</td><td class="n">.66</td><td class="n">.52</td><td class="n">.64</td><td class="gap">+.34</td></tr>
      <tr class="sep"><td>découpage (faux positifs)</td><td>COMET</td><td class="n">.76</td><td class="n">.82</td><td class="n">.89</td><td class="n">.85</td><td class="gap">+.06</td></tr>
      <tr><td>découpage (faux positifs)</td><td>CometKiwi</td><td class="n">.68</td><td class="n">.67</td><td class="n">.85</td><td class="n">.76</td><td class="gap">−.01</td></tr>
      <tr><td>structurelles (3)</td><td>les deux</td><td class="n">.90–1.00</td><td class="n">.87–1.00</td><td class="n">—</td><td class="n">—</td><td class="gap">≈ 0</td></tr>
    </tbody>
  </table>
  </div>
  <p class="aside"><b>Ce qui concorde&nbsp;:</b> les perturbations structurelles à w=1, le découpage
  (y compris sa dérive vers plus de faux positifs quand w grandit), et les effectifs — 957 paires de
  cohérence lexicale et 1&nbsp;533 de connecteurs, contre 956 et 1&nbsp;532 dans la Table 2 du papier.
  <b>Ce qui ne concorde pas&nbsp;:</b> les trois catégories lexicales, pour les deux métriques, à toutes les
  fenêtres — nos valeurs sont 21 à 34 points au-dessus. Le résultat signature du papier, CometKiwi sous le
  hasard à w=1 puis remontant avec le contexte, n'apparaît pas chez nous.</p>
</section>""")

    a("""<section class="slide">
  <header><span class="eyebrow">Vérification</span><h2>Trois causes possibles, une seule à trancher</h2></header>
  <ol class="steps">
    <li><span class="n">01</span><span><b>Le traitement des documents non perturbés.</b> Nous ne gardons que
    les documents où la perturbation a effectivement mordu (<span class="mono">Levenshtein &gt; 0</span>). Si
    le papier garde toute la catégorie, les documents identiques comptent comme des échecs et tirent
    l'exactitude vers le bas. C'est l'hypothèse la plus économique : elle explique l'ampleur et le sens de
    l'écart, mais pas exactement la même correction pour COMET et CometKiwi.</span></li>
    <li><span class="n">02</span><span><b>La version publiée diffère de celle du papier.</b> C'est établi
    pour deux catégories : le README annonce 1&nbsp;929 paires de permutation contre 1&nbsp;290 mesurées, et
    700 de découpage contre 118. Si les sous-ensembles lexicaux ont bougé de même, les exactitudes
    bougent.</span></li>
    <li><span class="n">03</span><span><b>Une différence d'agrégation.</b> Traitement des documents plus
    courts que w, pondération par paire plutôt que par document, ou micro/macro-moyenne entre paires de
    langues.</span></li>
  </ol>
  <p class="aside"><b>Le diagnostic qui tranche&nbsp;:</b> recalculer notre exactitude en gardant tous les
  documents de la catégorie, les paires identiques comptant comme des échecs, et voir si les colonnes
  lexicales rejoignent le papier. Une seule ligne à changer dans
  <span class="mono">build_units()</span> — mais il faut relancer l'évaluation sur le cluster, les écarts
  par document ne sont pas conservés dans le JSON.</p>
</section>""")

    # ── 9–10. Q3 ────────────────────────────────────────────
    a(slide_fig(7, "q3_duel", "Blocs FLORES+ : le duel à candidats fixes",
                "Erreur du duel à 5 candidats en fonction du nombre de phrases par bloc, et part des blocs éligibles",
                "Le bloc correct contre exactement 5 négatifs à une erreur, moyenné sur tous les tirages. "
                "<b>L'encodeur COMET est au hasard dès k=2</b> ; LaBSE et E5 restent exploitables. À droite, "
                "la contrepartie : exiger 5 négatifs écarte 86 % des blocs à k=2."))

    a(slide_fig(8, "q3_detection", "Le même verdict, sans sélection de blocs",
                "Taux de détection en fonction du nombre de phrases par bloc",
                "Duel à 1 candidat, couverture 100 %. <b>Le ré-entraînement gagne +.02 à +.03</b> — positif "
                "dans les 16 cellules, mais loin de restaurer la sensibilité perdue."))

    a("</div>\n<div id=\"counter\"></div>\n<div id=\"hint\">← → pour naviguer</div>")
    a(SCRIPT)

    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
