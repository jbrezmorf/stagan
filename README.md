# stagan
STAG analysis 

Analýza efektivity výuky.
- čtení STAGu přes REST API
- čtení programů, oborů, předmětů, rozvrhových akcí a studentů na nich
- čtení studentů na rozvrhových akcích ve vláknech, cca 20 tis. RA v jednotkách minut času
- memoizace čtení STAGU s použitím joblib
- čtení bodování předmětů podle odučených hodin dle rozpočtu FM v daném roce
- souhrnný graf zahrnující:
  - efektivní příjem předmětu na odučenou hodinu
  - průměry FMa ústavů přes předměty a zobrazené období
  - pro předměty graficky skladba fakult pro které se učí
- report s popisem zobrazených dat
- celé export z HTML do PDF

Nedostatky:
- v rozpočtech FM chybí některé předměty které se v daném roce učily, bylo by dobré částečně automatizovat 
  přípravu bodového ohodnocení a mít případné ruční opravy ve strojově čitelné podobě a s odůvodněním

- výstup by bylo lépe vizualizovat přes Jupyter notebook s volbami:
  - zahrnuté roky
  - čáry pro FM, ústavy, programy
  - lepší popis významu grafu
  - další grafy

