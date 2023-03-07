package travelplan.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
@AllArgsConstructor
public class TripInfo {

    private String startingPlace;

    private String endPlace;

    private Date departureTime;

    public TripInfo() {
        //Default Constructor
    }
}
