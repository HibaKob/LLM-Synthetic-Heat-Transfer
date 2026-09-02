"""
Use this script to create the human-generated seed examples .json file. The script writes the 
examples in the unified format, computes the ground-truth answers, verifies every example with
verification.py, and only then writes the JSON. Edit THIS file, not the JSON.

Workflow:
  1. Add Type A problems via add_type_a(...): you provide a check_expression, the
     script evaluates it to fill answer_numeric.
  2. Add Type B problems via add_type_b(...): you provide a reference_solution and
     test-case args; the script runs the reference to fill each expected value.
  3. Run:  python build_seeds.py
     It verifies all seeds and writes heat_transfer_seeds.json only if 100% pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from syntheticheatransfer import verification as v  # deterministic verifier


# Collected seeds
_seeds: list[dict] = []

###############################################################################
# Helper functions
###############################################################################

def add_type_a(
    id: str,
    topic: str,
    subtopic: str,
    instruction: str,
    reasoning: str,
    check_expression: str,
    answer_units: str,
    answer_tolerance_rel: float = 0.01,
    provenance: str = "human_authored",
) -> None:
    """
    Add an analytical (numeric-answer) problem.

    Provide check_expression (a pure Python expression using only `math` or `numpy`) that
    computes the answer. The script evaluates it to fill answer_numeric, so the
    stored answer is correct by construction.
    """
    answer_numeric = float(eval(check_expression, v.expression_namespace()))  # author-controlled expression
    _seeds.append({
        "id": id,
        "type": "A",
        "topic": topic,
        "subtopic": subtopic,
        "provenance": provenance,
        "instruction": instruction,
        "reasoning": reasoning,
        "answer_numeric": answer_numeric,
        "answer_units": answer_units,
        "answer_tolerance_rel": answer_tolerance_rel,
        "check_expression": check_expression,
    })


def add_type_b(
    id: str,
    topic: str,
    subtopic: str,
    instruction: str,
    function_name: str,
    reference_solution: str,
    cases: list[dict],
    provenance: str = "human_authored",
) -> None:
    """
    Add a coding problem.

    `cases` is a list of dicts, each: {"name": str, "args": list, "tol_rel": float}.
    The script runs reference_solution on each case's args to fill "expected", so
    expected values are exact and never hand-typed.

    Functions that return a single comparable value (scalar or a flat list) are preferred. 
    For long time-histories, return a summary value (e.g. final temp) to avoid huge 
    expected arrays.
    """
    fn = v.load_solution(reference_solution, function_name)
    test_cases = []
    for c in cases:
        args = list(c["args"])
        expected = fn(*args)
        test_cases.append({
            "name": c["name"],
            "args": args,
            "expected": expected,
            "tol_rel": float(c.get("tol_rel", 1e-6)),
        })
    _seeds.append({
        "id": id,
        "type": "B",
        "topic": topic,
        "subtopic": subtopic,
        "provenance": provenance,
        "instruction": instruction,
        "function_name": function_name,
        "reference_solution": reference_solution,
        "test_cases": test_cases,
    })


###############################################################################
# AUTHOR SEEDS   
###############################################################################

def author_seeds() -> None:

    # ---- Example Type A #1 ----
    add_type_a(
        id="cond-A-01",
        topic="conduction",
        subtopic="plane wall, Fourier's law",
        instruction=(
            "A plane wall of thickness 0.3 m has thermal conductivity 1.2 W/m*K. "
            "The inner surface is at 25 C and the outer surface at 38 C. For steady "
            "one-dimensional conduction, find the heat flux through the wall in W/m^2."
        ),
        reasoning=(
            "Steady 1-D conduction follows Fourier's law: q = -k*(T2 - T1)/L. "
            "q = -1.2*(25-38)/0.3."
        ),
        check_expression="-1.2*(25-38)/0.3",
        answer_units="W/m^2",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #2 ----
    add_type_a(
        id="cond-A-02",
        topic="conduction",
        subtopic="plane wall, heat rate with area",
        instruction=(
            "A wall of thermal conductivity 0.6 W/m*K is 6 m wide, 3 m high, and 0.2 m thick. "
            "The surface temperatures are 25 C and 15 C. Find the steady heat transfer rate through "
            "the wall in W."
        ),
        reasoning=(
            "Steady 1-D conduction follows Fourier's law: q = -k*(T2 - T1)/L. "
            "Cross-sectional surface area of the wall perpendicular to heat flow: A = H*W "
            "Heat transfer rate: Q = q*A. "
            "A = 6*3 = 18 m^2. "
            "Q = -0.6*18*(15-25)/0.2."
        ),
        check_expression="-0.6*18*(15-25)/0.2",
        answer_units="W",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #3 ----
    add_type_a(
        id="cond-A-03",
        topic="conduction",
        subtopic="thermal resistance (R-value), U-value",
        instruction=(
            "A wall has total thermal resistance (per unit area) of 1.2 m^2*K/W. If the "
            "temperature difference across it is 20 C, find the heat flux in W/m^2."
        ),
        reasoning=(
            "Steady-state heat transfer: q = deltaT/R = 20/1.2."
        ),
        check_expression="20/1.2",
        answer_units="W/m^2",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #4 ----
    add_type_a(
        id="cond-A-04",
        topic="conduction",
        subtopic="composite wall, series resistance",
        instruction=(
            "A composite wall consists of 2 layers in series. The first layer is 0.1 m thick "
            "and has thermal conductivity 0.8 W/m*K. The second layer is 0.03 m thick and has "
            "thermal conductivity 2.5 W/m*K. The surface temperatures on the outer faces of the " 
            "wall are 35 C and -8 C. Find the heat flux for a unit area in W/m^2."
        ),
        reasoning=(
            "Series thermal resistances (per unit area): R = L1/k1 + L2/k2. "
            "R = 0.1/0.8 + 0.03/2.5 = 0.137 m^2*K/W. "
            "Steady-state heat transfer (per unit area): q = -(T2-T1)/R. "
            "q = -(-8-35)/0.137."
        ),
        check_expression="-(-8-35)/0.137",
        answer_units="W/m^2",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #5 ----
    add_type_a(
        id="cond-A-05",
        topic="conduction",
        subtopic="cylindrical shell",
        instruction=(
            "A cast steel pipe (hollow cylinder, k = 50 W/m*K) has inner radius 0.04 m, "
            "10 mm wall thickness, and 1 m length. The inner surface is at 100 C, while "
            "the outer is at 40 C. Find the radial heat transfer rate in W."
        ),
        reasoning=(
            "Outer radius (r2) = inner radius (r1) + wall thickness. "
            "r2 = 0.04 + 10*10^-3 = 0.05 m."
            "Steady-state radial heat flow: Q = (2*pi*L*k(T1-T2))/ln(r2/r1). " 
            "Q = (2*pi*1*50*(100-40))/ln(0.05/0.04)."
        ),
        check_expression="(2*math.pi*1*50*(100-40))/math.log(0.05/0.04)",
        answer_units="W",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #6 ----
    add_type_a(
        id="cond-A-06",
        topic="conduction",
        subtopic="thermal resistance network with convection",
        instruction=(
            "A plane wall (thickness 0.1 m, k = 1.0 W/m*K, area 1 m^2) has hot fluid on "
            "one side at 150 C with convection coefficient 50 W/m^2*K, and cold fluid on "
            "the other side at 30 C with convection coefficient 20 W/m^2*K. Find the "
            "steady heat transfer rate in W."
        ),
        reasoning=(
            "Total resistance: R = 1/(h1*A) + L/(k*A) + 1/(h2*A). "
            "R = 1/(50*1.0) + 0.1/(1.0*1.0) + 1/(20*1.0) = 0.02 + 0.1 + 0.05 = 0.17 K/W. "
            "Steady-state heat transfer rate: Q = -(30-150)/(1/(50*1.0) + 0.1/(1.0*1.0) + 1/(20*1.0))." 
        ),
        check_expression="-(30-150)/(1/(50*1.0) + 0.1/(1.0*1.0) + 1/(20*1.0))",
        answer_units="W",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #7 ----
    add_type_a(
        id="cond-A-07",
        topic="conduction",
        subtopic="thermal resistance network with convection",
        instruction=(
            "A composite wall consists of 3 layers. The first (inner) layer is 4 mm thick and is "
            "made of plastic (k = 0.25 W/m*K). The outer layer (1.5 mm thick) consists of stainless "
            "steel with thermal conductivity 20 W/m*K. In between these two layers is an insulation"
            "layer of k = 0.07 W/m*K and width 0.2 m. The internal air temperature is -20 C when"
            "the external air temperature is 25 C, and the internal and external heat transfer "
            "coefficients are 12 W/m^2*K and 8.0 W/m^2*K respectively. Find the convective heat "
            "loss for a unit area in W/m^2."
        ),
        reasoning=(
            "Total thermal resistance (per unit area): R = 1/h_i + L_p/k_p + L_n/k_n + L_s/k_s + 1/h_o. "
            "R = 1/12 + (4*10^-3)/0.25 + 0.2/0.07 + (1.5*10^-3)/20 + 1/8.0 = 3.08 K*m^2/W. "
            "Steady-state heat transfer rate: Q = -(-20-25)/(1/12 + (4*1e-3)/0.25 + 0.2/0.07 + (1.5*1e-3)/20 + 1/8.0)." 
        ),
        check_expression="-(-20-25)/(1/12 + (4*1e-3)/0.25 + 0.2/0.07 + (1*1e-3)/20 + 1/8.0)",
        answer_units="W/m^2",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #8 ----
    add_type_a(
        id="cond-A-08",
        topic="conduction",
        subtopic="cylindrical shell, thermal resistance network with convection",
        instruction=(
            "A stainless steel pipe (k = 16 W/m*K) of length 50 m has an inner radius of 0.047 m "
            "and an outer radius of 0.05 mm respectively. The outer layer of the pipe is insulated "
            "with material of radial thickness 50 mm and k = 0.1 W/m*K. A fluid at temperature 80 C"
            "passes through the pipe while the outer surface is surrounded with air at 20C. The "
            "convective heat transfer coefficient due to water is 2000 W/m^2*K while that due to air "
            "is 200 W/m^2*K."
        ),
        reasoning=(
            "r3 = 0.05 + 50*10^-3 = 0.10 m."
            "Total thermal resistance: R = 1/(2*pi*r1*L*h_i) + ln(r2/r1)/(2*pi*L*k_pipe) + ln(r3/r2)/(2*pi*L*k_ins) + 1/(2*pi*r3*L*h_o). "
            "R = 1/(2*pi*0.047*50*2000) + ln(0.05/0.047)/(2*pi*50*16) + ln(0.1/0.05)/(2*pi*50*0.1) + 1/(2*pi*0.1*50*200) = 0.0223 K/W. "
            "Steady-state heat transfer rate: Q = -(20-80)/(1/(2*math.pi*0.047*50*2000) + math.log(0.05/0.047)/(2*math.pi*50*16) + "
            "math.log(0.1/0.05)/(2*math.pi*50*0.1) + 1/(2*math.pi*0.1*50*200))." 
        ),
        check_expression="-(20-80)/(1/(2*math.pi*0.047*50*2000) + math.log(0.05/0.047)/(2*math.pi*50*16) + math.log(0.1/0.05)/(2*math.pi*50*0.1) + 1/(2*math.pi*0.1*50*200))",
        answer_units="W",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #9 ----
    add_type_a(
        id="conv-A-01",
        topic="convection",
        subtopic="Newton's law of cooling",
        instruction=(
            "A flat plate of surface area 0.5 m^2 at 80 C is cooled by air at 20 C with a convection "
            "coefficient of 25 W/m^2*K. Find the convective heat loss in W."
        ),
        reasoning=(
            "Q = h*A*(Ts - Tinf) = 25*0.5*(80-20)"
        ),
        check_expression="25*0.5*(80-20)",
        answer_units="W",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #10 ----
    add_type_a(
        id="conv-A-02",
        topic="convection",
        subtopic="solve for convection coefficient",
        instruction=(
            "A flat plate of surface area 2 m^2 loses 1200 W by convection to its surroundings. The surface "
            "is at 60 C and the fluid is at 20 C. Find the convection coefficient h in W/m^2*K."
        ),
        reasoning=(
            "h = Q/(A*(Ts - Tinf)) = 1200/(2*(60-20))"
        ),
        check_expression="1200/(2*(60-20))",
        answer_units="W/m^2*K",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #11 ----
    add_type_a(
        id="conv-A-03",
        topic="convection",
        subtopic="Nusselt number to h",
        instruction=(
            "For flow over a plate, the Nusselt number is 100, the fluid thermal conductivity is 0.026 W/m*K, "
            "and the characteristic length is 0.5 m. Find the convection coefficient h in W/m^2*K."
        ),
        reasoning=(
            "Nu = (h*L)/k then h = (Nu*k)/L = (100*0.026)/0.5."
        ),
        check_expression="(100*0.026)/0.5",
        answer_units="W/m^2*K",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #12 ----
    add_type_a(
        id="conv-A-04",
        topic="convection",
        subtopic="Reynolds number",
        instruction=(
            "Air flows over a plate at velocity 10 m/s. The plate length is 2 m. Air kinematic "
            "viscosity is 1.5*10^-5 m^2/s. Find the Reynolds number at the trailing edge."
        ),
        reasoning=(
            "Re = (U*L)/nu = (10*2)/1.5e-5."
        ),
        check_expression="(10*2)/1.5e-5",
        answer_units="dimensionless",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

  # ---- Example Type A #13 ----
    add_type_a(
        id="conv-A-05",
        topic="convection",
        subtopic="Grashof number",
        instruction=(
            "A vertical rectangular plate is 0.5 m tall and maintained at a uniform " 
            "surface temperature of 80 C. It is exposed to ambient room air at 20 C. "
            "Calculate the Grashof number at the top edge of the plate, knowing that the "
            "kinematic viscosity of air at 50 C is 1.82*10^-5 m2/s. "
        ),
        reasoning=(
            "Film temperature: Tf = (Ts + Tinf)/2 = (80+20)/2 = 50 C."
            "Tf (in Kelvin) = Tf (in C) + 273.15 = 323.15 K."
            "Volumetric thermal expansion coefficient, for an ideal gas: beta = 1/Tf (in Kelvin) = 1/323.15 = 0.003095 1/K." \
            "Gr = (g*beta*(Ts - Tinf)*L^3)/nu^2 = (9.81*0.003095*(80-20)*(0.5)^3)/(1.82*10^-5)^2"
        ),
        check_expression="(9.81*(1/323.15)*(80-20)*(0.5)**3)/(1.82*1e-5)**2",
        answer_units="dimensionless",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

  # ---- Example Type A #14 ----
    add_type_a(
        id="conv-A-06",
        topic="convection",
        subtopic="Newton's law of cooling",
        instruction=(
            "A pipe of diameter 30 mm carries hot water. The external surface of the pipe is "
            "at 80 C and is subjected to a convective heat transfer coefficient of h = 6 W/m^2*K. "
            "Find the heat loss per meter length of the pipe due to convection to the surroundings "
            "which are at 15 C."
        ),
        reasoning=(
            "Q = h*A*(Ts-Tinf) = 6*(2*pi*(30/2)*10^-3*L)*(80-15)." 
            "Q/L = 6*(2*pi*(30/2)*10^-3)*(80-15)"
        ),
        check_expression="6*(2*math.pi*(30/2)*1e-3)*(80-15)",
        answer_units="W/m",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

  # ---- Example Type A #15 ----
    add_type_a(
        id="conv-A-07",
        topic="convection",
        subtopic="Combined conduction-convection surface temperature",
        instruction=(
            "A plane wall (area 1 m^2) conducts 400 W. Its outer surface "
            "loses heat to air at 25 C by convection with h = 40 W/m^2*K. Find the outer surface "
            "temperature in C."
        ),
        reasoning=(
            "At steady state, the conducted heat equals the convected heat." 
            "Q = h*A*(Ts - Tinf) then Ts = Q/(h*A) + Tinf = 400/(40*1) + 25"
        ),
        check_expression="400/(40*1) + 25",
        answer_units="C",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

  # ---- Example Type A #16 ----
    add_type_a(
        id="conv-A-08",
        topic="convection",
        subtopic="Natural convection",
        instruction=(
            "A long horizontal uninsulated pipe of diameter 0.2 m and length 20 m has an outside "
            "temperature of 75 C and sits in surrounding air at 25 C. Find the total heat transfer rate "
            "out of the pipe. Given the following air properties at 50 C: "
            "Kinematic viscosity (nu) = 1.82*10^-5 m^2/s, Thermal conductivity (k) = 0.0274 W/m*K, "
            "and Prandtl number (Pr) = 0.723."
        ),
        reasoning=(
            "Film temperature: Tf = (Ts + Tinf)/2 = (75+25)/2 = 50 C, "
            "Tf (Kelvin) = Tf(C)+273.15 = 50+273.15 = 323.15." 
            "Volumetric thermal expansion coefficient, for an ideal gas: beta = 1/Tf (Kelvin) = 1/323.15 = 0.003095 1/K."
            "Grashof number: Gr = (g*beta*(Ts - Tinf)*L^3)/nu^2 = (9.81*0.003095*(75-25)*(0.2)^3)/(1.82*10^-5)^2 = 3.67*10^7."
            "Rayleigh Number: Ra = Gr*Pr = (3.67*10^7)*0.723 = 2.65*10^7 < 10^9, the flow is laminar." 
            "Nu = [0.60+(0.387*Ra^(1/6))/([1+(0.559/Pr)^(9/16)]^(8/27))]^2, "
            "Nu = [0.60+(0.387*(2.65*10^7)^(1/6))/([1+(0.559/0.723)^(9/16)]^(8/27))]^2 = 37.89."
            "Heat transfer coefficient: h = (Nu*k)/D = (37.89*0.0274)/0.2 = 5.19 W/m^2*K."
            "Surface area: A = pi*D*L = pi*(0.2)*(20)"
            "Calculate heat loss using Newton's Law of Cooling: Q = h*A*(Ts-Tinf), "
            "Q = ((((0.60+(0.387*(((9.81*(1/323.15)*(75-25)*(0.2)^3)/(1.82*10^-5)^2)*0.723)^(1/6))/((1+(0.559/0.723)^(9/16))^(8/27)))^2)*0.0274)/0.2)*(pi*0.2*20)*(75-25)."
        ),

        check_expression="((((0.60+(0.387*(((9.81*(1/323.15)*(75-25)*(0.2)**3)/(1.82*1e-5)**2)*0.723)**(1/6))/((1+(0.559/0.723)**(9/16))**(8/27)))**2)*0.0274)/0.2)*(math.pi*0.2*20)*(75-25)",
        answer_units="W",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

 # ---- Example Type A #17 ----
    add_type_a(
        id="conv-A-09",
        topic="convection",
        subtopic="Natural convection",
        instruction=(   
            "A square horizontal flat plate measuring 0.3 m x 0.3 m is heated to a uniform "
            "temperature of 60 C. It is exposed to ambient air at 20 C. Calculate the rate "
            "of heat transfer from the upper surface of the plate. Given the following "
            "air properties at 40 C: Kinematic viscosity (nu) = 1.702*10^-5 m^2/s, "
            "Thermal conductivity (k) = 0.02662 W/m*K, and Prandtl number (Pr) = 0.7255."
        ),
        reasoning=(
            "Film temperature: Tf = (Ts + Tinf)/2 = (60+20)/2 = 40 C, "
            "Tf (Kelvin) = Tf(C)+273.15 = 40+273.15 = 313.15."
            "Volumetric thermal expansion coefficient, for an ideal gas: beta = 1/Tf (Kelvin) = 1/313.15 = 0.003193 1/K."
            "Characteristic length (Lc): Lc = A/P = (0.3)^2/(4*0.3) = 0.075 m." 
            "Grashof number: Gr = (g*beta*(Ts - Tinf)*Lc^3)/nu^2 = (9.81*0.003193*(60-20)*0.075^3)/(1.702*10^-5)^2 = 1.8247*10^6." 
            "Rayleigh Number: Ra = Gr*Pr =(1.8247*10^6)*0.7255 = 1.3238*10^6, 10^4 <= Ra <= 10^7, thus laminar flow."
            "Nusselt number, heat transfer from the upper surface of the plate: Nu = 0.54*Ra^(1/4) = 0.54*(1.3238*10^6)^(1/4) = 18.32."
            "Heat transfer coefficient: h = (Nu*k)/Lc = (18.32*0.02662)/0.075 = 6.502 W/m^2*K."
            "Rate of heat transfer: Q = h*A*(Ts-Tinf) = (((0.54*(((9.81*(1/313.15)*(60-20)*0.075^3)/(1.702*10^-5)^2 )*0.7255)^(1/4))*0.02662)/((0.3)^2/(4*0.3)))*(0.3)^2*(60-20)"
        ),
        check_expression="(((0.54*(((9.81*(1/313.15)*(60-20)*0.075**3)/(1.702*1e-5)**2 )*0.7255)**(1/4))*0.02662)/((0.3)**2/(4*0.3)))*(0.3)**2*(60-20)",
        answer_units="W",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

  # ---- Example Type A #18 ----
    add_type_a(
        id="conv-A-10",
        topic="convection",
        subtopic="Forced convection",
        instruction=(
            "A horizontal thin plate (L = 25 mm, W = 8 mm) is electrically heated such that "
            "its surface temperature reaches 40 C. Wind (T = 20 C, Cp = 1.005 kJ/kg*K, "
            "nu = 1.522*10^-5 m^2/s, rho = 1.19 kg/m^3, Pr = 0.72) blows parallel to the "
            "longest side of the plate at a velocity uinf = 10 m/s. Calculate the heat dissipated "
            "from both sides of the plate."
        ),
        reasoning=(
            "Check flow the regime by evaluating the Reynolds number at the trailing edge of the plate: " 
            "Re = (uinf*L)/nu = (10*(25*10^-3))/(1.522*10^-5) = 16425.76 < 5*10^5, this means we have "
            "laminar flow across the entire plate."
            "Thermal conductivity of the air: Pr = (nu*rho*Cp)/k then k = (nu*rho*Cp)/Pr, "
            "k = (1.522*10^-5*1.19*1.005*10^3)/0.72 = 0.02528 W/m*K." \
            "Average Nusselt number using the correlation for laminar flow over the entire "
            "isothermal flat plate: Nu = 0.664*Re^(1/2)*Pr^(1/3) = 0.664*16425.76^(1/2)*0.72^(1/3) = 76.274."
            "Average heat transfer coefficient: Nu = (h*L)/k then h = (Nu*k)/L, "
            "h = (76.274*0.02528)/(25*10^-3)=  77.128 W/m^2*K."
            "Since heat is dissipated from both sides of the plate, the total area Atotal = 2*A, "
            "Atotal = 2*(25*10^-3)*(8*10^-3) = 0.0004 m^2."
            "Total heat dissipation: Q = h*Atotal*(Ts-Tinf), "
            "Q = (((0.664*((10*(25*10^-3))/(1.522*10^-5))**(1/2)*0.72**(1/3))*((1.522*10^-5*1.19*1.005*10^3)/0.72))/(25*10^-3))*(2*(25*10^-3)*(8*10^-3))*(40-20)" 
        ),
        check_expression="(((0.664*((10*(25*1e-3))/(1.522*1e-5))**(1/2)*0.72**(1/3))*((1.522*1e-5*1.19*1.005*1e3)/0.72))/(25*1e-3))*(2*(25*1e-3)*(8*1e-3))*(40-20)",
        answer_units="W",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

  # ---- Example Type A #19 ----
    add_type_a(
        id="conv-A-11",
        topic="convection",
        subtopic="Forced convection, surface temperature",
        instruction=(
            "A vertical plate having the dimensions 0.3 m x 0.2 m dissipates "
            "100 W from one side. Air at Tinf = 30 C (Cp = 1 kJ/kg*K, k = 0.03 W/m*K, "
            "mu = 2*10^-5 kg/m*s, R = 287 J/kg*K, P = 1 bar) is blown at a speed uinf = 12 m/s "
            "parallel to the longest dimension of the plate. Find the surface temperature "
            "of the plate. Consider a critical Reynolds number of 10^5."
        ),
        reasoning=(
            "Ideal gas law to find the density of air: rho = P/(R*Tinf), "
            "rho=(10^5)/(287*(30+273.15)) = 1.149 kg/m^3."
            "Check flow the regime by evaluating the Reynolds number: Re_L = (rho*uinf*L)/mu,"
            "Re_L = (1.149*12*0.3)/(2*10^-5) = 2.068*10^5, since 10^5 < Re < 3*10^5, the flow is "
            "mixed, that is transitioning from laminar to turbulent."
            "Prandtl number: Pr = (mu*Cp)/k = ((2*10^-5)*(1*10^3))/0.03 = 0.6667." 
            "Average Nusselt number for a mixed boundary layer using the following relations: "
            "Nu = 0.332*Re^0.5*Pr^(1/3) for Re < 10^5 and Nu = 0.037*Re^0.8*Pr^(1/3) for Re >= 10^5,"
            "Nu_avg = (h_avg*L)/k, Re_x = (rho*u*x)/mu, h_avg = (1/L)*(int|0,xL| h_lam*dx + int|xL,L| h_turb*dx) "
            "We get: Nu_avg = (0.04625*(Re_L)^0.8-252.5)*Pr^(1/3), then "
            "Nu_avg = (0.04625*(2.068*10^5)^0.8-252.5)*0.6667^(1/3) = 501.96."
            "Average convection coefficient: h_avg = (Nu_avg*k)/L = (501.96*0.03)/0.3 = 50.196 W/m^2*K."
            "Area of plate: A = L*W = 0.3*0.2 = 0.06 m^2"
            "Surface temperature: Q = h*A*(Ts-Tinf), then Ts = Q/(h*A)+Tinf = 100/(50.196*0.06)+30"
        ),
        check_expression="100/(((((0.04625*(2.068*1e5)**0.8-252.5)*0.6667**(1/3))*0.03)/0.3)*(0.3*0.2))+30",
        answer_units="C",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #20 ----
    add_type_a(
        id="rad-A-01",
        topic="radiation",
        subtopic="Stefan-Boltzmann emissive power, black body",
        instruction=(
            "A black surface is at 500 K. Find its blackbody emissive power in W/m^2. "
            "Use Stefan-Boltzmann constant, sigma = 5.67*10^-8 W/m^2*K^4."
        ),
        reasoning=(
            "Eb = sigma*T^4 =(5.67*10^-8)*500^4."
        ),
        check_expression="5.67e-8*500**4",
        answer_units="W/m^2",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #21 ----
    add_type_a(
        id="rad-A-02",
        topic="radiation",
        subtopic="Gray surface net radiation to surroundings",
        instruction=(
            "A gray surface of emissivity 0.8, area 1.5 m^2, at 400 K radiates to large "
            "surroundings at 300 K. Find the net radiative heat transfer in W. "
            "Use sigma = 5.67*10^-8 W/m^2*K^4."
        ),
        reasoning=(
            "Q = epsilon*sigma*A*(Ts^4 - Tsur^4) = 0.8*(5.67*10^-8)*1.5*(400^4 - 300^4)."
        ),
        check_expression="0.8*(5.67*1e-8)*1.5*(400**4 - 300**4)",
        answer_units="W",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #22 ----
    add_type_a(
        id="rad-A-03",
        topic="radiation",
        subtopic="View factor",
        instruction=(
            "Two adjacent flat discs (Surface 1 and Surface 2) each of 0.4 m diameter "
            "are enclosed in a 0.1 m wide protective ring casing (Surface 3). Given that "
            "the view factor between surfaces 1 and 2 (F12) is equal to 0.6, calculate F31."
        ),
        reasoning=(
            "Surface 1 is a flat disc. Flat surfaces cannot see themselves. Thus, we have F11 = 0."
            "Summation rule in an enclosure: F11 + F12 + F13 = 1, then F13 = 1 - (F11 + F12) = 1 - (0 + 0.6) = 0.4." 
            "Reciprocity rule: Ai*Fij = Aj*Fji then A1*F13 = A3*F31 and F31 = (A1/A3)*F13."
            "A1 = pi*D^2/4 = pi*(0.4)^2/4 = 0.04*pi and A3 = pi*D*L = pi*(0.4)*(0.1) = 0.04*pi."
            "Since A1 = A3, F31 = F13 = 0.4."
        ),
        check_expression="1 - (0 + 0.6)",
        answer_units="dimensionless",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #23 ----
    add_type_a(
        id="rad-A-04",
        topic="radiation",
        subtopic="Radiosity, Two-surface enclosure",
        instruction=(
            "Consider an enclosure formed by two infinitely long, parallel plates of equal area (A1 = A2 = 1 m^2) "
            "separated by an angle. Surface 1 is maintained at T1 = 800 K with an emissivity epsilon1 = 0.6. "
            "Surface 2 is maintained at T2 = 400 K with an emissivity epsilon2 = 0.3. The geometric view "
            "factor from Surface 1 to Surface 2 is F12 = 0.5. Find the radiosity J1. Use sigma = 5.67*10^-8 W/m^2*K^4."
        ),
        reasoning=(
            "Blackbody emissive power: Eb = sigma*T^4 then Eb1 = (5.67*10^-8)*(800)^4 and Eb2 = (5.67*10^-8)*(400)^4, " \
            "Eb1 = 23224.32 W/m^2 and Eb2 = 1451.52 W/m^2."
            "Radiation network resistances: Rtotal = R1 + R12 + R2, "
            "Surface resistance R1 = (1-epsilon1)/(epsilon1*A1) = (1-0.6)/(0.6*1) = 2/3 m^(-2), "
            "Surface resistance R2 = (1-epsilon2)/(epsilon2*A2) = (1-0.3)/(0.3*1) = 7/3 m^(-2), " 
            "Space resistance R12 = 1/(A1*F12) = 1/(1*0.5) = 2 m^(-2), then "
            "Rtotal = 2/3 + 2 + 7/3 = 5 m^(-2)." 
            "Net radiation heat transfer Q12 = (Eb1 - Eb2)/Rtotal = (23224.32 - 1451.52)/5 = 4354.56 W."
            "Net heat transfer leaving Surface 1 can be expressed in terms of its blackbody power and its radiosity: " \
            "Q12 = (Eb1-J1)/R1 then J1 = Eb1 - Q12*R1 = 23224.32 - 4354.56*(2/3)"
        ),
        check_expression="(5.67*1e-8)*(800)**4-(((5.67*1e-8)*(800**4-400**4))/((1-0.6)/(0.6*1)+(1-0.3)/(0.3*1)+1/(1*0.5)))*((1-0.6)/(0.6*1))",
        answer_units="W/m^2",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #24 ----
    add_type_a(
            id="rad-A-05",
            topic="radiation",
            subtopic="Radiosity, three-surface enclosure",
            instruction=(
                "An infinitely long cylinder (Surface 1) is enclosed symmetrically by two flat "
                "plates forming a 90-degree corner box (Surface 2 and Surface 3). The dimensions "
                "per unit length are structured to give the following surface areas: "
                "Surface 1 (Cylinder): A1 = 1.0 m^2, Surface 2 (Plate 1): A2 = 2.0 m^2, "
                "Surface 3 (Plate 2): A3 = 2.0 m^2. Find the radiosity J1 for the following thermal "
                "conditions: Surface 1: T1 = 1000 K, epsilon1 = 0.8, "
                "Surface 2: T2 = 600 K, epsilon2 = 0.5, Surface 3: T3 = 300 K, epsilon3 = 0.7. " 
                "Use sigma = 5.67*10^-8 W/m^2*K^4."
                
            ),
            reasoning=(
                "Blackbody emissive power: Eb = sigma*T^4, Eb1 = sigma*T1^4, Eb2 = sigma*T2^4, Eb3 = sigma*T3^4, "
                "then Eb1 = (5.67*10^-8)*(1000)^4 = 56700 W/m^2, Eb2 = (5.67*10^-8)*(600)^4 = 7348.32 W/m^2, "
                "Eb3 = (5.67*10^-8)*(300)^4 = 459.27 W/m^2."
                "View factors: Cylinder is fully enclosed by the two symmetrical plates and it cannot see itself, " 
                "hence, F11 = 0. The cylinder splits its view equally between the two sides, therefore F12 = F13 = 0.5. "
                "Surfaces 1 and 2 are flat plates then F22 = 0 and F33 = 0." 
                "Reciprocity rule (Ai*Fij = Aj*Fji): A1*F12 = A2*F21 then F21 = (A1/A2)*F12 = (1/2)*0.5 = 0.25, "
                "and F31 = (A1/A3)*F13 = (1/2)*0.5 = 0.25"
                "Summation rule in an enclosure: F21 + F22 + F23 = 1 then F23 = 1 - (F21 + F22) = 1-(0.25+0) = 0.75, "
                "and F31 + F32 + F33 = 1 then F32 = 1 - (F31 + F33) = 1-(0.25+0) = 0.75."
                "Linear System for Radiosities: Ji - (1 - epsiloni)*sum(1 to N)Fij*Jj = epsiloni*Ebi then, "
                "For surface 1, J1 - (1 - epsilon1)*(F11*J1 + F12*J2 + F13*J3) = epsilon1*Eb1, then J1 - (1 - 0.8)*(0*J1 + 0.5*J2 + 0.5*J3) = 0.8*5.67*10^-8)*(1000)^4 "
                "For surface 2, J2 - (1 - epsilon2)*(F21*J1 + F22*J2 + F23*J3) = epsilon2*Eb2, then J2 - (1 - 0.5)*(0.25*J1 + 0*J2 + 0.75*J3) = 0.5*(5.67*10^-8)*(600)^4 "
                "For surface 3, J3 - (1 - epsilon3)*(F31*J1 + F32*J2 + F33*J3) = epsilon3*Eb3, then J3 - (1 - 0.7)*(0.25*J1 + 0.75*J2 + 0*J3) = 0.7*(5.67*10^-8)*(300)^4 "
                "Matrix formulation: Jm = [J1, J2, J3] = Inv(A)*C where A = [1 -(1 - 0.8)*0.5 -(1 - 0.8)*0.5; -(1 - 0.5)*0.25 1 -(1 - 0.5)*0.75; -(1 - 0.7)*0.25 -(1 - 0.7)*0.75 1] "
                "and C = [0.8*5.67*10^-8)*(1000)^4; 0.5*(5.67*10^-8)*(600)^4; 0.7*(5.67*10^-8)*(300)^4]." \
                "J1 = Jm[0] = Inv([1 -(1 - 0.8)*0.5 -(1 - 0.8)*0.5; -(1 - 0.5)*0.25 1 -(1 - 0.5)*0.75; -(1 - 0.7)*0.25 -(1 - 0.7)*0.75 1])*[0.8*5.67*10^-8)*(1000)^4; 0.5*(5.67*10^-8)*(600)^4; 0.7*(5.67*10^-8)*(300)^4][0]"
            ),
            check_expression="np.linalg.inv(np.array([[1, -(1 - 0.8)*0.5, -(1 - 0.8)*0.5],[-(1 - 0.5)*0.25, 1.0, -(1 - 0.5)*0.75],[-(1 - 0.7)*0.25, -(1 - 0.7)*0.75,  1.0  ]])).dot(np.array([(0.8*5.67*1e-8)*(1000)**4, 0.5*(5.67*1e-8)*(600)**4, 0.7*(5.67*1e-8)*(300)**4]))[0]",
            answer_units="W/m^2",
            answer_tolerance_rel=0.02,
            provenance="human_authored",
        )

    # ---- Example Type A #25 ----
    add_type_a(
        id="fin-A-01",
        topic="extended_surfaces",
        subtopic="Fin parameter m",
        instruction=(
            "A pin fin has convection coefficient h = 50 W/m^2*K, perimeter P = 0.02 m, "
            "thermal conductivity k = 200 W/m*K, and cross-sectional area Ac = 2.5*10^-5 m^2. "
            "Find the fin parameter m."
        ),
        reasoning=(
            "m = sqrt((h*P)/(k*Ac)) = sqrt((50*0.02)/(200*2.5e-5))."
        ),
        check_expression="math.sqrt((50*0.02)/(200*2.5e-5))",
        answer_units="1/m",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #26 ----
    add_type_a(
        id="fin-A-02",
        topic="extended_surfaces",
        subtopic="Infinite fin heat rate",
        instruction=(
            "An infinitely long fin has a cross-sectional area of 3*10^-5 m^2 and a perimeter of 0.03 m. "
            "The fin is made of a material with thermal conductivity k = 180 W/m*K. The base of the fin is "
            "maintained at a steady temperature of 120 C while it is exposed to ambient air at temperature 25 C "
            "with a convective heat transfer coefficient of 40 W/m^2*K. Calculate the rate of heat transfer from the fin."
        ),
        reasoning=(
            "For an infinitely long fin: Q = sqrt(h*P*k*Ac)*(Tb-Tinf) = sqrt(40*0.03*180*3*10^-5)*(120-25)"
        ),
        check_expression="math.sqrt(40*0.03*180*3*1e-5)*(120-25)",
        answer_units="W",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #27 ----
    add_type_a(
        id="fin-A-03",
        topic="extended_surfaces",
        subtopic="Short fin heat rate",
        instruction=(
            "An aluminum pin fin with a length of 0.06 m, a perimeter of 0.04 m, and a cross-sectional area of "
            "5*10^-5 m^2 extends from a surface at 150 C. The thermal conductivity of the aluminum is 205 W/m*K. "
            "The fin is exposed to ambient air at 20 C with a heat transfer coefficient of 55 W/m^2*K. Calculate "
            "the rate of heat transfer from the fin."
        ),
        reasoning=(
            "Corrected length: Lc = L + Ac/P = 0.06+(5*10^-5)/0.04 = 0.06125 m."
            "Fin parameter: m = sqrt((h*P)/(k*Ac)) = sqrt((55*0.04)/(205*5*10^-5)) = 14.65 1/m."
            "Baseline maximum heat transfer: M = sqrt(h*P*k*Ac)*(Tb-Tinf) = sqrt(55*0.04*205*5*10^-5)*(150-20) = 19.52 W."
            "Adjustment for short fin: Q = M*tanh(m*Lc) = 19.52*tanh(14.65*0.06125)."
        ),
        check_expression="(math.sqrt(55*0.04*205*5*1e-5)*(150-20))*math.tanh(math.sqrt((55*0.04)/(205*5*1e-5))*(0.06+(5*1e-5)/0.04))",
        answer_units="W",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #28 ----
    add_type_a(
        id="fin-A-04",
        topic="extended_surfaces",
        subtopic="Fin efficiency, finite length",
        instruction=(
            "A cooling fin with length 5 cm has a cross-sectional area of 6*10^-5 m^2 "
            "and a perimeter of 0.05 m. The fin material has a thermal conductivity of 230 W/m*K "
            "and is attached to a base wall maintained at a steady temperature of 220 C. The fin "
            "extends into an ambient air environment at 25 C with a convective heat transfer "
            "coefficient of 40 W/m^2*K. Assuming a perfectly insulated (adiabatic) tip, "
            "calculate the fin efficiency (eta) for this setup."
        ),
        reasoning=(
            "Fin parameter: m = sqrt((h*P)/(k*Ac)) = sqrt((40*0.05)/(230*6*10^-5)) = 12.04 1/m."
            "Fin efficiency: eta = tanh(m*L)/(m*L) = tanh(12.04*(5*10^-2))/(12.04*(5*10^-2))"
        ),
        check_expression="math.tanh((math.sqrt((40*0.05)/(230*6*1e-5)))*(5*1e-2))/((math.sqrt((40*0.05)/(230*6*1e-5)))*(5*1e-2))",
        answer_units="dimensionless",
        answer_tolerance_rel=0.001,
        provenance="human_authored",
    )

    # ---- Example Type A #29 ----
    add_type_a(
        id="trans-A-01",
        topic="transient",
        subtopic="Lumped capacitance Biot number",
        instruction=(
            "A metal sphere of diameter 0.02 m has k = 50 W/m*K and is cooled with "
            "h = 30 W/m^2*K. Using characteristic length Lc = V/As = D/6, compute the "
            "Biot number and state whether lumped-capacitance analysis is valid (Bi < 0.1)."
        ),
        reasoning=(
            "Lc = D/6 = 0.02/6 = 0.003333 m. "
            "Bi = (h*Lc)/k = (30*0.003333)/50 = 0.002. Since Bi < 0.1, lumped capacitance is valid."
        ),
        check_expression="(30*(0.02/6))/50",
        answer_units="dimensionless",
        answer_tolerance_rel=0.05,
        provenance="human_authored",
    )

    # ---- Example Type A #30 ----
    add_type_a(
        id="trans-A-02",
        topic="transient",
        subtopic="Lumped capacitance time constant",
        instruction=(
            "A lumped-capacitance object has density 8000 kg/m^3, specific heat 500 J/kg*K, "
            "volume 10^-4 m^3, surface area 0.02 m^2, thermal conductivity 35 W/m*K and "
            "convection coefficient 100 W/m^2*K. Find the thermal time constant (tau)."
        ),
        reasoning=(
            "Lc = V/As = (10^-4)/0.02 = 0.005 m."
            "Bi = (h*Lc)/k = (100*0.005)/35 = 0.0143. Since Bi < 0.1, lumped capacitance is valid."
            "Thermal time constant in a lumped-capacitance system: tau = (rho*V*c)/(h*As) = (8000*10^-4*500)/(100*0.02)"
        ),
        check_expression="(8000*1e-4*500)/(100*0.02)",
        answer_units="s",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #31 ----
    add_type_a(
        id="trans-A-03",
        topic="transient",
        subtopic="Lumped capacitance temperature at time t",
        instruction=(
            "A lumped object with time constant 200 s is at 300 C in an environment "
            "at 20 C. Find its temperature after 200 s."
        ),
        reasoning=(
            "T(t) = Tinf + (T0 - Tinf)*exp(-t/tau),"
            "At t = tau = 200 s, T(tau) = 20 + (300-20)*exp(-tau/tau) = 20+(300-20)*exp(-1)"
        ),
        check_expression="20+(300-20)*math.exp(-1)",
        answer_units="C",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #32 ----
    add_type_a(
        id="trans-A-04",
        topic="transient",
        subtopic="Thermal diffusivity",
        instruction=(
            "A material has thermal conductivity 40 W/m*K, density 2000 kg/m^3, "
            "and specific heat 800 J/kg*K. Find its thermal diffusivity alpha."
        ),
        reasoning=(
            "alpha = k/(rho*cp) = 40/(2000*800)."
        ),
        check_expression="40/(2000*800)",
        answer_units="m^2/s",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #33 ----
    add_type_a(
        id="trans-A-05",
        topic="transient",
        subtopic="Fourier number",
        instruction=(
            "A slab has thermal diffusivity 10^-5 m^2/s and characteristic length 0.05 m. "
            "Find the Fourier number after 500 s."
        ),
        reasoning=(
            "Fo = (alpha*t)/Lc^2 = (10^-5*500)/(0.05)^2"
        ),
        check_expression="(1e-5*500)/(0.05)**2",
        answer_units="dimensionless",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #34 ----
    add_type_a(
        id="hx-A-01",
        topic="heat_exchangers",
        subtopic="LMTD counterflow",
        instruction=(
            "In a counterflow heat exchanger, the hot fluid enters at 120 C and "
            "exits at 80 C; the cold fluid enters at 30 C and exits at 70 C. "
            "Find the log mean temperature difference (LMTD) in C."
        ),
        reasoning=(
            "For counterflow: dT1 = Th_in - Tc_out = 120-70 = 50; "
            "dT2 = Th_out - Tc_in = 80-30 = 50. "
            "Since dT1 = dT2, LMTD = dT1 = 50 C (limit of the log-mean formula)."
        ),
        check_expression="120-70",
        answer_units="C",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #35 ----
    add_type_a(
        id="hx-A-02",
        topic="heat_exchangers",
        subtopic="LMTD parallel flow",
        instruction=(
            "In a parallel-flow heat exchanger, hot fluid enters at 150 C, cold at 40 C; "
            "hot exits at 90 C, cold exits at 70 C. Compute the LMTD in C."
        ),
        reasoning=(
            "Parallel flow: dT1 = Th_in - Tc_in = 150-40 = 110; "
            "dT2 = Th_out - Tc_out = 90-70 = 20. "
            "LMTD = (110-20)/ln(110/20) = 90/ln(5.5)."
        ),
        check_expression="(110-20)/math.log(110/20)",
        answer_units="C",
        answer_tolerance_rel=0.02,
        provenance="human_authored",
    )

    # ---- Example Type A #36 ----
    add_type_a(
        id="hx-A-03",
        topic="heat_exchangers",
        subtopic="Heat duty from LMTD",
        instruction=(
            "A heat exchanger has overall coefficient U = 500 W/m^2*K, area 4 m^2, "
            "and LMTD = 40 C. Find the heat transfer rate in W."
        ),
        reasoning=(
            "Q = U*A*LMTD = 500*4*40."
        ),
        check_expression="500*4*40",
        answer_units="W",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #37 ----
    add_type_a(
        id="hx-A-04",
        topic="heat_exchangers",
        subtopic="Energy balance mass flow",
        instruction=(
            "Water (cp = 4180 J/kg*K) is heated from 25 C to 55 C in a heat exchanger "
            "that delivers 62700 W. Find the required water mass flow rate in kg/s."
        ),
        reasoning=(
            "Q = m_dot*cp*deltaT,"
            "m_dot = Q/(cp*deltaT) = 62700/(4180*(55-25))"
        ),
        check_expression="62700/(4180*(55-25))",
        answer_units="Kg/s",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #38 ----
    add_type_a(
        id="energy-A-01",
        topic="energy_balance",
        subtopic="Sensible heat",
        instruction=(
            "How much energy is required to raise the temperature of 5 kg of aluminum (cp = 900 J/kg*K) " 
            "from 20 C to 100 C? Give the answer in J."
        ),
        reasoning=(
            "Q = m*cp*deltaT = 5*900*(100-20) = 5*900*80"
        ),
        check_expression="5*900*80",
        answer_units="J",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type A #39 ----
    add_type_a(
        id="energy-A-02",
        topic="energy_balance",
        subtopic="Heating power and time",
        instruction=(
            "A 2000 W heater warms 10 kg of water (cp = 4180 J/kg*K) from 20 C to 60 C. " 
            "Assuming no losses, how long does it take in seconds?"
        ),
        reasoning=(
            "Energy needed E = m*cp*deltaT = 10*4180*(60-20) = 1672000 J. "
            "Time t = E/P = 1672000/2000."
        ),
        check_expression="(10*4180*(60-20))/2000",
        answer_units="s",
        answer_tolerance_rel=0.01,
        provenance="human_authored",
    )

    # ---- Example Type B #1 ----
    add_type_b(
        id="cond-B-01",
        topic="conduction",
        subtopic="1-D steady conduction, finite difference",
        instruction=(
            "Write a Python function solve_1d_conduction(L, n, T_left, T_right, k) that "
            "solves steady-state 1-D heat conduction with no heat generation in a rod of "
            "length L using the finite difference method with n equally spaced nodes "
            "(including both boundaries). Return a list of the n node temperatures. With "
            "no source term the exact solution is a linear temperature profile."
        ),
        function_name="solve_1d_conduction",
        reference_solution=(
            "import numpy as np\n"
            "def solve_1d_conduction(L, n, T_left, T_right, k):\n"
            "    A = np.zeros((n, n)); b = np.zeros(n)\n"
            "    A[0,0]=1.0; b[0]=T_left; A[-1,-1]=1.0; b[-1]=T_right\n"
            "    for i in range(1, n-1):\n"
            "        A[i,i-1]=1.0; A[i,i]=-2.0; A[i,i+1]=1.0; b[i]=0.0\n"
            "    return list(np.linalg.solve(A, b))"
        ),
        cases=[
            {"name": "linear_100_to_0", "args": [1, 11, 100, 0, 1], "tol_rel": 1e-6},
            {"name": "linear_50_to_150", "args": [2, 5, 50, 150, 10], "tol_rel": 1e-6},
        ],
        provenance="human_authored",
    )

    # ---- Example Type B #2 ----
    add_type_b(
        id="cond-B-02",
        topic="conduction",
        subtopic="1-D conduction with uniform generation",
        instruction=(
            "Write a Python function solve_conduction_with_source(L, n, T_left, T_right, k, q_gen) "
            "that solves steady 1-D conduction with uniform volumetric heat generation q_gen (W/m^3) "
            "using finite differences with n nodes. Return the n node temperatures. The governing "
            "equation is k*d2T/dx2 + q_gen = 0."
        ),
        function_name="solve_conduction_with_source",
        reference_solution=(
            "import numpy as np\n"
            "def solve_conduction_with_source(L, n, T_left, T_right, k, q_gen):\n"
            "   dx = L/(n-1)\n"
            "   A = np.zeros((n, n)); b = np.zeros(n)\n"
            "   A[0,0] = 1.0; b[0] = T_left\n"
            "   A[-1,-1] = 1.0; b[-1] = T_right\n"
            "   for i in range(1, n-1):\n"
            "       A[i, i-1] = 1.0; A[i, i] = -2.0; A[i, i+1] = 1.0\n"
            "       b[i] = -q_gen*dx*dx/k\n"
            "   return list(np.linalg.solve(A, b))"
        ),
        cases=[
            {"name": "symmetric_parabola", "args": [1, 101, 0, 0, 1, 2], "tol_rel": 1e-6},
            {"name": "nonzero_bcs", "args": [1, 51, 20, 40, 2, 100], "tol_rel": 1e-6},
        ],
        provenance="human_authored",
    )

    # ---- Example Type B #3 ----
    add_type_b(
        id="conv-B-01",
        topic="extended_surfaces",
        subtopic="Infinite fin temperature profile",
        instruction=(
            "Write a Python function fin_temp_profile(L, n, m, T_base, T_inf) that returns the "
            "temperature distribution along an infinitely-long fin, evaluated at n equally spaced "
            "points from x=0 to x=L. The analytical profile is (T - T_inf) = (T_base - T_inf)*exp(-m*x). "
            "Return a list of n temperatures."
        ),
        function_name="fin_temp_profile",
        reference_solution=(
            "import math\n"
            "def fin_temp_profile(L, n, m, T_base, T_inf):\n"
            "   temps = []\n"
            "   for i in range(n):\n"
            "       x = L*i/(n-1)\n"
            "       theta = (T_base - T_inf)*math.exp(-m*x)\n"
            "       temps.append(T_inf + theta)\n"
            "   return temps"
        ),
        cases=[
            {"name": "decay_profile", "args": [0.2, 5, 14.14, 120, 25], "tol_rel": 1e-6},
        ],
        provenance="human_authored",
    )

    # ---- Example Type B #4 ----
    add_type_b(
        id="hx-B-01",
        topic="heat_exchangers",
        subtopic="LMTD computation",
        instruction=(
            "Write a Python function lmtd_hx(dT1, dT2) that returns the log mean temperature difference "
            "for a heat exchanger given the two end temperature differences dT1 and dT2. Handle the special "
            "case dT1 = dT2 (where the formula's limit equals dT1). Formula: LMTD = (dT1 - dT2)/ln(dT1/dT2)."
        ),
        function_name="lmtd_hx",
        reference_solution=(
            "import math\n"
            "def lmtd_hx(dT1, dT2):\n"
            "   ans = dT1\n"
            "   if abs(dT1 - dT2) < 1e-9:\n"
            "       ans = dT1\n"
            "   else:\n"
            "       ans = (dT1 - dT2)/math.log(dT1/dT2)\n"
            "   return ans"
        ),
        cases=[
            {"name": "equal_differences", "args": [50, 50], "tol_rel": 1e-9},
            {"name": "parallel_flow", "args": [110, 20], "tol_rel": 1e-6},
            {"name": "large_ratio", "args": [100, 10], "tol_rel": 1e-6},
        ],
        provenance="human_authored",
    )

    # ---- Example Type B #5 ----
    add_type_b(
        id="trans-B-01",
        topic="transient",
        subtopic="l1-D transient conduction, explicit FTCS",
        instruction=(
            "Write a Python function transient_1d(L, n, alpha, T_init, T_left, T_right, t_end, dt) that "
            "solves the 1-D transient heat equation dT/dt = alpha*d2T/dx2 using the explicit forward-time "
            "centered-space (FTCS) scheme. The rod has n nodes, uniform initial temperature T_init, "
            "and fixed boundary temperatures T_left and T_right. Return the final temperature profile "
            "as a list of n values. Note the stability limit: alpha*dt/dx^2 <= 0.5."
        ),
        function_name="transient_1d",
        reference_solution=(
            "def transient_1d(L, n, alpha, T_init, T_left, T_right, t_end, dt):\n"
            "   dx = L/(n-1)\n"
            "   r = alpha*dt/(dx*dx)\n"
            "   T = [T_init]*n\n"
            "   T[0] = T_left; T[-1] = T_right\n"
            "   t = 0.0\n"
            "   while t < t_end - 1e-9:\n"
            "       Tn = T[:]\n"
            "       for i in range(1, n-1):\n"
            "           Tn[i] = T[i] + r*(T[i+1] - 2*T[i] + T[i-1])\n"
            "       Tn[0] = T_left; Tn[-1] = T_right\n"
            "       T = Tn; t += dt\n"
            "   return T"
        ),
        cases=[
            {"name": "approach_steady_state", "args": [1, 11, 0.0001, 0, 100, 0, 100000.0, 0.4], "tol_rel": 1e-3},
        ],
        provenance="human_authored",
    )

    # ---- Example Type B #6 ----
    add_type_b(
        id="trans-B-02",
        topic="transient",
        subtopic="lumped capacitance ODE integration (forward Euler)",
        instruction=(
            "Write a Python function lumped_cooling(T0, T_inf, tau, t_end, dt) that computes the "
            "temperature history of a lumped-capacitance object using explicit forward Euler integration "
            "of dT/dt = -(T - T_inf)/tau. Integrate from t=0 to t_end in steps of dt, starting from T0. "
            "Return a list of temperatures at each step, including the initial value at t=0. The analytical "
            "solution is T(t) = T_inf + (T0 - T_inf)*exp(-t/tau)."
        ),
        function_name="lumped_cooling",
        reference_solution=(
            "def lumped_cooling(T0, T_inf, tau, t_end, dt):\n"
            "   temps = [T0]; t = 0.0; T = T0\n"
            "   while t < t_end - 1e-9:\n"
            "       T = T + dt*(-(T - T_inf)/tau)\n"
            "       t = t + dt\n"
            "       temps.append(T)\n"
            "   return temps"
        ),
        cases=[
            {"name": "cool_100_to_20", "args": [100, 20, 50, 50, 0.01], "tol_rel": 1e-6},
            {"name": "heat_up_20_to_80", "args": [20, 80, 30, 30, 0.01], "tol_rel": 1e-6},
        ],
        provenance="human_authored",
    )

###############################################################################
# Build + verify + write
###############################################################################

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "seeds" / "heat_transfer_seeds.json"


def build() -> None:

    v.create_output_folder(REPO_ROOT, "data")           
    v.create_output_folder(DATA_DIR, "seeds")

    author_seeds()

    if not _seeds:
        raise SystemExit("No seeds authored. Add problems in author_seeds().")

    # Check for duplicate ids
    ids = [s["id"] for s in _seeds]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"Duplicate seed ids: {sorted(dupes)}")

    # Verify every seed
    summary = v.verify_batch(_seeds)
    print(f"Seeds: {summary['total']} | passed: {summary['passed']} | failed: {summary['failed']}")
    failed = [r for r in summary["results"] if not r.passed]
    for r in failed:
        print(f"  FAIL [{r.problem_id}] ({r.problem_type}): {r.detail}")

    if failed:
        raise SystemExit("Not all seeds verified. Fix the failing seeds; JSON not written.")

    # Provenance report (transparency)
    from collections import Counter
    prov = Counter(s["provenance"] for s in _seeds)
    print("Provenance:", dict(prov))

    # Coverage report (encourages spread)
    topics = Counter(s["topic"] for s in _seeds)
    types = Counter(s["type"] for s in _seeds)
    print("Topics:", dict(topics))
    print("Types:", dict(types))

    out = {
        "domain": "heat_transfer",
        "description": (
            "Seed problems for LLM-based synthetic data generation. Built and verified "
            "by `build_seeds.py`. Type A = analytical reasoning with a numeric final "
            "answer (rule-based verification). Type B = coding problem, solution is a "
            "Python function verified by execution against known cases."
        ),
        "seeds": _seeds,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAll seeds verified. Wrote {OUTPUT_PATH} ({len(_seeds)} seeds).")


if __name__ == "__main__":
    build()