package rebook.entity;

import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
public class TripAllDetailInfo {

    private String tripId;

    private Date travelDate;

    private String from;

    private String to;

    public TripAllDetailInfo() {
        //Default Constructor
    }

}
