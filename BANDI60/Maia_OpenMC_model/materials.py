import openmc


#MATERIALS FILE
"""CREATE FUEL - UO2"""
uo2 = openmc.Material(material_id=1, name="$\mathrm{UO_2}$", temperature=900)
uo2.add_nuclide("U235", 0.0495, percent_type="ao")  # 4.95%
uo2.add_nuclide("U238", 0.9505, percent_type="ao")  # 95.05%
uo2.add_element("O", 2.0, percent_type="ao")
uo2.set_density("g/cm3", 10.76)    #source found


"""GAP"""     #FIND SOURCE
helium = openmc.Material(material_id=2, name="helium", temperature=900)
helium.add_element("He", 1.0)
helium.set_density("g/cm3", 2.449e-3) #source found

"""CLADDING""" #FIND SOURCE
zirc = openmc.Material(material_id=3, name="zircaloy4", temperature=600)
zirc.add_element("Zr", 98.070, percent_type="wo")
zirc.add_element("Sn", 1.500, percent_type="wo")
zirc.add_element("Fe", 0.220, percent_type="wo")
zirc.add_element("Cr", 0.130, percent_type="wo")
zirc.add_element("O", 0.100, percent_type="wo")
zirc.set_density("g/cm3", 6.551)     #source found and calculated

"""CREATE MODERATOR"""
water = openmc.Material(material_id=4, name="water", temperature=600)
water.add_nuclide("H1", 2.0, percent_type="ao")
water.add_nuclide("O16", 1.0, percent_type="ao")
water.add_s_alpha_beta("c_H_in_H2O")
water.set_density("g/cm3", 0.687)    #source found

"""CREATE PYREX"""
SiO2 = openmc.Material(material_id=5, name="SiO2", temperature=600)
SiO2.add_element("Si", 1.0, percent_type="ao")
SiO2.add_element("O", 2.0, percent_type="ao")
SiO2.set_density("g/cm3", 2.200) #source found and calculated

K2O = openmc.Material(material_id=6, name="K2O", temperature=600)
K2O.add_element("K", 2.0, percent_type="ao")
K2O.add_element("O", 1.0, percent_type="ao")
K2O.set_density("g/cm3", 2.312) #source found and calculated

Al2O3 = openmc.Material(material_id=7, name="Al2O3", temperature=600)
Al2O3.add_element("Al", 2.0, percent_type="ao")
Al2O3.add_element("O", 3.0, percent_type="ao")
Al2O3.set_density("g/cm3", 3.981) #source found and calculated

B2O3 = openmc.Material(material_id=8, name="B2O3", temperature=600)
B2O3.add_element("B", 2.0, percent_type="ao")
B2O3.add_element("O", 3.0, percent_type="ao")
B2O3.set_density("g/cm3", 2.231) #source found and calculated

"""MIX PYREX"""
pyrex5 = openmc.Material.mix_materials([SiO2, K2O, Al2O3, B2O3], [0.8735, 0.0437, 0.0328, 0.0500], percent_type="wo", name="pyrex5")
pyrex10 = openmc.Material.mix_materials([SiO2, K2O, Al2O3, B2O3], [0.8276, 0.0414, 0.0310, 0.1000], percent_type="wo", name="pyrex10")
pyrex25 = openmc.Material.mix_materials([SiO2, K2O, Al2O3, B2O3], [0.6896, 0.0345, 0.0259, 0.2500], percent_type="wo", name="pyrex25")
pyrex35 = openmc.Material.mix_materials([SiO2, K2O, Al2O3, B2O3], [0.5977, 0.0299, 0.0224, 0.3500], percent_type="wo", name="pyrex35")
pyrex40 = openmc.Material.mix_materials([SiO2, K2O, Al2O3, B2O3], [0.5517, 0.0275, 0.0208, 0.4000], percent_type="wo", name="pyrex40")

pyrex5.temperature = 600 #type: ignore
pyrex5.set_density("g/cm3", 2.265)

pyrex10.temperature = 600 #type: ignore
pyrex10.set_density("g/cm3", 2.263)

pyrex25.temperature = 600 #type: ignore
pyrex25.set_density("g/cm3", 2.258)

pyrex35.temperature = 600 #type: ignore
pyrex35.set_density("g/cm3", 2.254)

pyrex40.temperature = 600 #type: ignore
pyrex40.set_density("g/cm3", 2.252)



"""SHIELDING"""
#SA-508 STEEL for RPV - source found
steel = openmc.Material(material_id=14, name="Steel_SA508")
steel.add_element("Fe", 96.897, "wo")
steel.add_element("B", 0.003, "wo")
steel.add_element("V", 0.05, "wo")
steel.add_element("Cu", 0.35, "wo")
steel.add_element("Mo", 0.08, "wo")
steel.add_element("Cr", 0.25, "wo")
steel.add_element("Ni", 0.25, "wo")
steel.add_element("S", 0.035, "wo")
steel.add_element("P", 0.035, "wo")
steel.add_element("Si", 0.60, "wo")
steel.add_element("Mn", 1.20, "wo")
steel.add_element("C", 0.25, "wo")
steel.set_density("g/cm3", 7.850)   #room T
steel.temperature = 293.6 #type: ignore

#biological shielding - still have no info on this??

#1.4404 STEEL for CV - source found
steel2 = openmc.Material(material_id=15, name="Steel_1.4404")
steel2.add_element("Fe", 65.505, "wo")
steel2.add_element("N", 0.11, "wo")
steel2.add_element("Ni", 11.5, "wo")
steel2.add_element("Mo", 2.25, "wo")
steel2.add_element("Cr", 17.5, "wo")
steel2.add_element("S", 0.015, "wo")
steel2.add_element("P", 0.045, "wo")
steel2.add_element("Mn", 2.0, "wo")
steel2.add_element("Si", 1.0, "wo")
steel2.add_element("C", 0.03, "wo")
steel2.set_density("g/cm3", 8.000)  #room T
steel2.temperature = 293.6 #type: ignore


materials = openmc.Materials([uo2, helium, zirc, water, pyrex5, pyrex10, pyrex25, pyrex35, pyrex40, steel, steel2])
materials.export_to_xml()
