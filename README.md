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

wide table for naklady:
idx: year, katedra, zkratka, program, -> n_students, n_studenti_program, KEN_program ->  
wide table naklady:
idx: year, katedra, zkratka -> n_hod_pr, n_hod_cv, n_hod_sem, n_par_cv, body

- 12.11. 
  - predmety sebrane podle oboru nejsou kompletni, tj. predmet se podle studentu muze objevit i na 
    programu kde normalne neni; Pro rok 2024 je v rozpoctu pro KMA/GSW:  7 v prezencnim a 2 v Kombinovanem 
    na programu B0114A300064, zadny na programu B0114A300070 (to je tedy nespravne)
  - podle rozvrhu je v 2024 na xx64 11 studentů (z toho 5 Komb.), + 7 studentů jiných programů;  
  - aktualne pocitam pocty studentu na programu podle skutecne ucenych studentu, prg_id je primo z reportu studenta
  - to je patrně více než má rektorátní rozpočet

- 15.11.
  - rozsah obsahuje občas cifru s písmenem: 5T, 7T, 5S, 5D; objevuje se to jak u  Cv tak u Př   
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