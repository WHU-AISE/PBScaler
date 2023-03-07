package preserve.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.ToString;

/**
 * @author fdse
 */
@Data
@AllArgsConstructor
@ToString
public class TripAllDetail {

    private boolean status;

    private String message;

    private TripResponse tripResponse;

    private Trip trip;

    public TripAllDetail() {
        //Default Constructor
    }

}
