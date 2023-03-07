package preserveOther.entity;

import lombok.Data;
import lombok.ToString;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import java.util.Date;

/**
 * @author fdse
 */
@Data
@ToString
public class TripResponse {
    @Valid
    private TripId tripId;

    @Valid
    @NotNull
    private String startingStation;

    @Valid
    @NotNull
    private String terminalStation;

    @Valid
    @NotNull
    private Date startingTime;

    @Valid
    @NotNull
    private Date endTime;

    /**
     * The number of economy seat
     */
    @Valid
    @NotNull
    private int economyClass;

    /**
     * The number of confort seat
     */
    @Valid
    @NotNull
    private int confortClass;

}
