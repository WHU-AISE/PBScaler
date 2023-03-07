package rebook.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class TripAllDetail {

    private TripResponse tripResponse;

    private Trip trip;

    public TripAllDetail() {
        //Default Constructor
    }

}
