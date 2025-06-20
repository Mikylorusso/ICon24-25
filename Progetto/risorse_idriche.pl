:- use_module(library(csv)).

% Carica il file CSV e genera i fatti con validazione dei dati
carica_risorse:-
    csv_read_file('risorse_idriche.csv', Rows, [functor(row)]),
    forall(
        member(row(Nazione, AcquaAnnuale, AcquaProCapite, Popolazione), Rows),
        (   valid_data(Nazione, AcquaAnnuale, AcquaProCapite, Popolazione) ->
            assertz(acqua_annuale(Nazione, AcquaAnnuale)),
            assertz(acqua_pro_capite(Nazione, AcquaProCapite)),
            assertz(popolazione(Nazione, Popolazione))
         ;  write('Dati non validi per: '), write(Nazione), nl
        )
    ).

% Validazione dei dati in ingresso
valid_data(_, AcquaAnnuale, AcquaProCapite, Popolazione):-
    number(AcquaAnnuale),
    number(AcquaProCapite),
    number(Popolazione).

% Calcola l'acqua totale utilizzata annualmente da una nazione (ottimizzato)
:- dynamic calcola_acqua/2.

acqua_totale_utilizzata(Nazione, AcquaTotale):-
    calcola_acqua(Nazione, AcquaTotale), !. % Usa il valore memorizzato
acqua_totale_utilizzata(Nazione, AcquaTotale):-
    acqua_annuale(Nazione, AcquaAnnuale),
    acqua_pro_capite(Nazione, AcquaProCapite),
    popolazione(Nazione, Popolazione),
    AcquaTotale is AcquaAnnuale + (AcquaProCapite * Popolazione),
    assertz(calcola_acqua(Nazione, AcquaTotale)). % Memorizza il risultato

% Previsione della domanda futura di acqua (fattore dinamico)
previsione_domanda_futura(Nazione, Fattore, DomandaFutura):-
    calcola_acqua(Nazione, Fattore, DomandaFutura).

% Calcolo generico con un fattore
calcola_acqua(Nazione, Fattore, Risultato):-
    acqua_totale_utilizzata(Nazione, AcquaTotale),
    Risultato is AcquaTotale * Fattore.

% Verifica sostenibilità
puo_soddisfare_futura(Nazione, Fattore):-
    previsione_domanda_futura(Nazione, Fattore, DomandaFutura),
    acqua_totale_utilizzata(Nazione, AcquaTotale),
    AcquaTotale >= DomandaFutura.

% Gestione della crisi idrica con soglia configurabile
crisi_idrica(Nazione, SogliaCrisi):-
    acqua_totale_utilizzata(Nazione, AcquaTotale),
    AcquaTotale =< SogliaCrisi,
    write('Crisi idrica in '), write(Nazione),
    write('. Iniziare azioni di emergenza!'), nl.

% Statistiche aggregate: acqua totale per tutte le nazioni
acqua_totale_globale(Totale):-
    findall(AcquaTotale, acqua_totale_utilizzata(_, AcquaTotale), ListaAcque),
    sum_list(ListaAcque, Totale).

% Raccomandazioni personalizzate
raccomandazioni(Nazione, SogliaCrisi):-
    (   crisi_idrica(Nazione, SogliaCrisi) ->
        write('Suggerimenti per '), write(Nazione), write(': '), nl,
        write('- Ridurre gli sprechi.'), nl,
        write('- Investire in infrastrutture per la conservazione dell''acqua.'), nl,
        write('- Promuovere l''uso sostenibile delle risorse.'), nl
    ;   write('La gestione delle risorse di '), write(Nazione),
        write(' è soddisfacente al momento.'),nl).
