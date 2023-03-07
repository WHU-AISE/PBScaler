package ticketinfo.service;

import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;
import ticketinfo.entity.Travel;

/**
 * Created by Chenjie Xu on 2017/6/6.
 */
public interface TicketInfoService {
    Response queryForTravel(Travel info, HttpHeaders headers);
    Response queryForStationId(String name,HttpHeaders headers);
}
