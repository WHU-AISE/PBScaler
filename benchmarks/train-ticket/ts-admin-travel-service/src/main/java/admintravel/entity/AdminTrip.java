package admintravel.entity;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author fdse
 */
@Data
@NoArgsConstructor
public class AdminTrip {
    private Trip trip;
    private TrainType trainType;
    private Route route;
}
