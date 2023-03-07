package travel.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class TripAllDetail {
    private TripResponse tripResponse;

    private Trip trip;

    public TripAllDetail() {
    }

    public TripAllDetail(TripResponse tripResponse, Trip trip) {
        this.tripResponse = tripResponse;
        this.trip = trip;
    }

}
