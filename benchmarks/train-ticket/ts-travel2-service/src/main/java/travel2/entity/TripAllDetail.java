package travel2.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class TripAllDetail {

    private boolean status;

    private String message;

    private TripResponse tripResponse;

    private Trip trip;

    public TripAllDetail() {
        //Default Constructor
    }

}
