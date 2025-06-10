

distances = { "avant": 0.45, "arriere": 1.20, "gauche": 0.6, "droite": 0.55}
distance_max= 0
direction_max=" "
for direction, distance in distances.items():
    
    if distance < 0.5 :
        print(f'{direction} :  {distance}  m !!! obstacle proche')
     
    print(f'{direction} :  {distance}  m')

    if distance > distance_max:
        distance_max=distance
        direction_max= direction
print(f'la zone la plus degagee : {direction_max}  ({distance_max}m)')