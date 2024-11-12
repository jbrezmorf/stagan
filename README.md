# stagan
STAG analysis 

Analýza efektivity výuky.
- čtení STAGu přes REST API
- logování: hlavní skript se zastaví a čeká na vstup, otevře přihlašovací stránku STAG web API, o přihlášení je uživatel přesměrován na neexistující stránku
  URL stránky je třeba zkopírovat a zadat jako vstup skriptu a odeslat pomocí Enter
- URL obsahuje přihlašovací token uživatele, ten je automaticky uložen a je platný po 30 minut
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

Implementation:
- The functional style is used in order to use effective persistent memoization using joblib.
- Functional means we prefere definition of dataclasses and functions transforming one data to other rather then modify them.

Files:
- `analysis_main.py` : main script with configuration of the analysis and orchestration of STAG scrapping,
   rozpocet_FM reading, calculation of derived data and calling plot and report functions
- `stag.py` : STAG scrapping interface using REST API, not complete, just used requests and 
  basic normalization is performed directly
- `tables.py` : some persisting dictionaries assumed by STAG: KENs, facoulty sohrtcuts, katedra -> facoulty
- `vyuka_plot.py` : plot functions for the collected data frame of teached `predmet` 
- `report.py` : compose plot with its description and save it to PDF