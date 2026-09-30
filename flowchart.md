```mermaid
graph TD
    A[Start] --> B[User Selects LAZ File]
    B --> C[Read File Header & Classes]
    C --> D[User Selects Classes, Angle, Grid Size, Max Dist]
    D --> E[Read Points matching Classes in Chunks]
    E --> F[Combine filtered Points into Memory]
    F --> G[Rotate Points by -Angle]
    
    subgraph Grid Generation
    G --> H[Calculate Bounding Box in Rotated Space]
    H --> I[Generate Axis-Aligned Grid (u, v)]
    I --> J[Rotate Grid Points back by +Angle (x, y)]
    end
    
    subgraph Interpolation
    F --> K[Build KDTree from Source Points]
    J --> L[Query Nearest Source Point for each Grid Point]
    L --> M{Distance < Max Dist?}
    M -- Yes --> N[Assign Z value from Source]
    M -- No --> O[Assign Z = 0.0]
    end
    
    N --> P[Write X, Y, Z to CSV]
    O --> P
    P --> Q[End / Success Message]
```
