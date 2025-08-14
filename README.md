# Preventing SARS-CoV-2 superspreading events with antiviral intranasal sprays
 Figure-generating code for the [Journal of Theoretical Biology article](https://doi.org/10.1016/j.jtbi.2025.112237), "Preventing SARS-CoV-2 superspreading events with antiviral intranasal sprays" by Booth, G et al.

## Reproducing figures
Each figure is produced by a series of Jupyter Notebooks:
* [Figure 2D](cruiseship/plot/figure2d.ipynb): Posterior predictive check of model simulation vs data.
* [Figure 3](cruiseship/plot/figure3.ipynb): The impact of efficacy, uptake and intervention start time in passangers and crew on the total number of infections
* [Figure 4A-C](conference/plot/figure4a_c.ipynb): SPHH contact network and weighted contact matrix
* [Figure 4E-H](conference/plot/figute4e_h.ipynb): Treatment adherence vs dosing regimen on the number of infections averted, and the reduction in individual risk

Click [here](cruiseship/models) for the Diamond cruise ship transmission models. <br />
Click [here](conference/models/conference_model.py) for the SPHH conference transmission model.
